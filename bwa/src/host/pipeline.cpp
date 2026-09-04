// pipeline for bwa-mem operation.
#include "pipeline.h"
#include "concurrentqueue.h"
#include <queue>
#include <deque>
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include "timer.h"
#include <string.h>
#include "macro.h"
#include "cuda_wrapper.h"

extern void printPoolPeaks(void);

#define MAX_RECORD_BYTES 1024
#define MAX_QUEUE_SIZE_FACTOR 8

extern float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS];
extern int bwa_align(int gpuid, process_data_t *proc, g3_opt_t *g3_opt);

std::atomic<int> dispatch_queue_size;
std::atomic<int> active_loader_cnt;
std::atomic<int> active_dispatcher_cnt;
std::atomic<int> active_samgen_cnt;
std::atomic<long long> loaded_cnt;
std::atomic<long long> dispatched_cnt;
std::atomic<long long> written_cnt;
std::mutex write_queue_mutex;

// Per-GPU SAM-gen task queue (background SAM gen overlaps with GPU compute).
// The task carries the slot's proc (for pinned h_out_* arrays) and a CUDA event
// that fires when the async D2H copies into those arrays are complete.
struct SamGenTask {
    aligned_chunk_t *ac;    // nullptr = shutdown sentinel
    process_data_t  *proc;  // slot whose h_out_* pinned arrays have output data
    void            *d2h_done; // opaque cudaEvent_t; fires when async D2H copies complete
};

struct SamGenQueue {
    std::deque<SamGenTask> q;
    std::mutex mu;
    std::condition_variable cv;
};
static SamGenQueue samgen_qs[MAX_NUM_GPUS];

void pipeline(pipeline_aux_t *aux)
{
	// pipeline queues
	moodycamel::ConcurrentQueue<parsed_chunk_t *> dispatch_queue;
	std::priority_queue<
		aligned_chunk_t *, std::vector<aligned_chunk_t *>, writequeue_compare
		> write_queue;
	loaded_cnt = dispatched_cnt = written_cnt = 0;
	dispatch_queue_size = 0;

	// load input from disk
	std::vector<std::thread> load_threads;
	if(aux->load_thread_cnt == 0){ // to test loading at GPU-zero envs.
		active_loader_cnt = aux->load_thread_cnt = 1;
	} else{
		active_loader_cnt = aux->load_thread_cnt;
	}
	for(int t=0; t<aux->load_thread_cnt; t++) {
		load_threads.emplace_back(worker_load_and_parse, 
				std::ref(aux->g3_opt),
				std::ref(aux->fd_input),
				std::ref(aux->load_chunk_bytes),
				t, aux->load_thread_cnt, std::ref(dispatch_queue));
	}

	// work (align) with GPUs
	int num_workers = aux->g3_opt->num_use_gpus;
	active_dispatcher_cnt = num_workers;
	active_samgen_cnt = num_workers;

	// one background SAM-gen thread per GPU, started before GPU threads so
	// the first batch can be picked up immediately after memcpy_output.
	std::vector<std::thread> samgen_threads;
	for (int t = 0; t < num_workers; t++)
		samgen_threads.emplace_back(worker_samgen, t, aux, std::ref(write_queue));

	auto gpu_worker = [&](int tid) {
		// Load index into buffer 0 first — this may use most available VRAM on large refs.
		memcpy_index(aux->proc[tid][0], tid, aux);

		// Try to allocate the second buffer for double-buffering.
		// Done AFTER index load: on hg38 the index alone is ~45 GB, leaving <4 GB
		// for a second intermediate buffer (~3.3 GB).  Allocating both buffers before
		// the index would leave insufficient room for the index itself.
		size_t free_vram = get_free_vram();
		// Headroom: pool_mb + measured intermediate VRAM (from first buffer) + 128 MB safety.
		size_t buf_needed = (size_t)aux->g3_opt->pool_mb * MB_SIZE
		                  + aux->proc[tid][0]->intermediate_bytes
		                  + 128ULL * MB_SIZE;
		if (free_vram >= buf_needed) {
			aux->proc[tid][1] = device_alloc(tid, aux);
			share_index(aux->proc[tid][1], aux->proc[tid][0]);
		} else {
			aux->proc[tid][1] = aux->proc[tid][0]; // single-buffer fallback
			std::cerr << "* GPU " << tid
			          << " double-buffer skipped (free VRAM "
			          << free_vram / MB_SIZE << " MB < "
			          << buf_needed / MB_SIZE << " MB needed)\n";
		}

		worker_dispatch(std::ref(dispatch_queue),
				tid,
				std::ref(aux),
				aux->proc[tid][0],
				aux->proc[tid][1],
				std::ref(write_queue));
	};
	std::vector<std::thread> gpu_threads;
	for (int t=0; t < num_workers; t++) {
		gpu_threads.emplace_back(gpu_worker, t);
	}

	// write SAM header before any alignment records
	{
		const bntseq_t *bns = aux->idx->bns;
		for (int i = 0; i < bns->n_seqs; i++)
			*aux->samout << "@SQ\tSN:" << bns->anns[i].name
			             << "\tLN:" << bns->anns[i].len << "\n";
		*aux->samout << "@PG\tID:g3sa\tPN:g3sa\tVN:1.0\n";
		aux->samout->flush();
	}

	// write sams to disk in-order
	std::thread write_thread(worker_write, std::ref(write_queue), std::ref(aux->samout),
			std::ref(aux->g3_opt));

	for (auto &th : load_threads)
		th.join();
	for (auto &th : gpu_threads)
		th.join();

	// Send shutdown sentinels to samgen threads (ac=nullptr signals exit).
	for (int t = 0; t < num_workers; t++) {
		std::lock_guard<std::mutex> lk(samgen_qs[t].mu);
		samgen_qs[t].q.push_back({nullptr, nullptr, nullptr});
		samgen_qs[t].cv.notify_one();
	}
	for (auto &th : samgen_threads)
		th.join();

	write_thread.join();
	for(int t=0; t<num_workers; t++){
		destruct_proc(aux->proc[t][0]);
		if (aux->proc[t][1] != nullptr && aux->proc[t][1] != aux->proc[t][0])
			destruct_proc(aux->proc[t][1]);
	}

	printPoolPeaks();
	std::cerr << "* ALL PROCESSING DONE\n";
	std::cerr << "* loaded cnt: " << loaded_cnt << "\n";
	std::cerr << "* dispatched cnt: " << dispatched_cnt << "\n";
	std::cerr << "* written cnt: " << written_cnt << "\n";
	return;
}


const uint8_t encode_nt4[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};



void worker_load_and_parse(
		g3_opt_t *g3_opt,
		int fd_input,
		long load_chunk_bytes,
		int tid,
		int thread_cnt,
		moodycamel::ConcurrentQueue<parsed_chunk_t *> &dispatch_queue
		)
{
	long long file_offset;
	ssize_t loaded_bytes, rec_loaded_bytes;
	std::string buffer;
	int niter;
	char *p, *q;

	LARGE_TIMER_INIT();
	TIMER_INIT();

	LARGE_TIMER_START();
	long long name_offset, qual_offset, seq_offset;
	char buf_record[MAX_RECORD_BYTES];
	int nline;
	niter = 0;
	int id;
	bool first = true;
	while(true) {
		if(g3_opt->bound_load_queue &&
				dispatch_queue_size >= MAX_QUEUE_SIZE_FACTOR * thread_cnt){
			continue;
		} 
		dispatch_queue_size++;
		if(first){
			TIMER_START();
		}
		parsed_chunk_t *pc = new parsed_chunk_t;
		id = tid + thread_cnt * niter++;
		file_offset = load_chunk_bytes * id;
		pc->chunk_offset = file_offset;


		long load_request_bytes = load_chunk_bytes + g3_opt->single_record_bytes;

		buffer.clear();
		buffer.resize(load_request_bytes);
		// load a chunk.
		loaded_bytes = pread(fd_input, &buffer[0], load_request_bytes,
				file_offset);
		fflush(stderr);
		if(loaded_bytes == 0) break; // EOF.

		char *begin, *end;
		begin = &buffer[0];
		end = &buffer[0] + load_chunk_bytes;

		// boundary calibration
		while(*begin != '@') begin++;

		if(loaded_bytes < load_chunk_bytes){ // at some EOF
			end = &buffer[0] + loaded_bytes;
		} else{
			while(end < &buffer[0] + loaded_bytes &&
					*end != '@') end++;
		}

		// parse the loaded chunk.
		// input format:
		// @NAME EXTRA\n
		// SEQ\n
		// EXTRA2\n
		// QUAL\n
		// (repeats).
		char *pos, *base;
		pos = base = begin;
		pc->chunk_size = 0;
		name_offset = qual_offset = seq_offset = 0;
		for(pos = begin; pos < end; pos++){ // always *pos = '@' or term. here.
			pc->name_offsets.emplace_back(name_offset);
			pc->seq_offsets.emplace_back(seq_offset);
			pc->qual_offsets.emplace_back(qual_offset);

			for(base = ++pos; pos < end && *pos != ' '; pos++) ; // NAME
			pc->name.append(base, pos - base);
			name_offset += pos - base;
			for(; pos < end && *pos != '\n'; pos++) ; // skip EXTRA
			for(base = ++pos; pos < end && *pos != '\n'; pos++) ; // SEQ
			for(char *c = base; c < pos; c++)
				pc->seq.emplace_back(encode_nt4[(int)*c]);
			seq_offset += pos - base;
			for(++pos; pos < end && *pos != '\n'; pos++) ; // skip EXTRA2
			for(base = ++pos; pos < end && *pos != '\n'; pos++) ; // QUAL
			pc->qual.append(base, pos - base);
			qual_offset += pos - base;

			pc->chunk_size++;
		}

		// 1 additional sentinel
		pc->name_offsets.emplace_back(name_offset);
		pc->qual_offsets.emplace_back(qual_offset);
		pc->seq_offsets.emplace_back(seq_offset);

		// enqueue
		loaded_cnt += pc->chunk_size;
		dispatch_queue.enqueue(pc);
		if(first){
			TIMER_END(0, "loaded the first chunk");
			tprof[tid][FILE_INPUT_FIRST] += (float)(duration.count() / 1000);
			first = false;
		}
	}
	LARGE_TIMER_END(1, "loaded all chunks");
	tprof[tid][FILE_INPUT] += (float)(large_duration.count() / 1000);

	active_loader_cnt--; // EOF
	return;
}

void worker_dispatch(
		moodycamel::ConcurrentQueue<parsed_chunk_t *> &dispatch_queue,
		int tid,
		pipeline_aux_t *aux,
		process_data_t *proc0,
		process_data_t *proc1,
		std::priority_queue<
		aligned_chunk_t *, std::vector<aligned_chunk_t *>, writequeue_compare
		> &write_queue
		)
{
	LARGE_TIMER_INIT();
	LARGE_TIMER_START();
	TIMER_INIT();
	process_data_t *procs[2] = {proc0, proc1};
	int cur = 0;
	int gpuid = tid;
	aligned_chunk_t *ac;
	parsed_chunk_t *pc;

	// o_stream for offsets-only copy (stream-specific sync, no device stall);
	// d2h_stream for 6 bulk arrays (overlaps with next batch's GPU compute).
	void *o_stream   = cuda_o_stream_create(gpuid);
	void *d2h_stream = cuda_d2h_stream_create(gpuid);

	// Pre-fetched batch state: if prefetch_valid, ac_pre/pc_pre are ready.
	aligned_chunk_t *ac_pre = nullptr;
	bool prefetch_valid = false;

	// Track cumulative read count to compute correct n_processed (batch_offset)
	// for the hash tiebreaker in sort_regions.  Passing batch_offset=0 would make
	// sort_regions use hash_64(seqID) instead of hash_64(n_processed+seqID),
	// diverging from bwa-mem2 for reads in batch > 0.
	int64_t n_reads_completed = 0;  // total reads for which bwa_align has finished

	while(true){
		// Use pre-fetched batch if available, otherwise dequeue.
		if (prefetch_valid) {
			ac = ac_pre;
			ac_pre = nullptr;
			prefetch_valid = false;
		} else if(dispatch_queue.try_dequeue(pc)){
			dispatch_queue_size--;
			TIMER_START();
			ac = new aligned_chunk_t;
			ac->chunk_offset = pc->chunk_offset;
			ac->chunk_size = pc->chunk_size;
			ac->seq_offsets = pc->seq_offsets;
			ac->seq = pc->seq;
			ac->name_offsets = pc->name_offsets;
			ac->name = pc->name;
			ac->qual_offsets = pc->qual_offsets;
			ac->qual = pc->qual;
			delete pc;
			dispatched_cnt += ac->chunk_size;
			TIMER_END(0, "dequeued a chunk");
		} else {
			if(active_loader_cnt == 0){
				break; // fastq all loaded.
			}
			continue;
		}

		TIMER_START();
		memcpy_input(ac->chunk_size, procs[cur],
				ac->seq.data(), ac->seq_offsets.data());
		TIMER_END(0, "sent a chunk");
		tprof[gpuid][PUSH_TOTAL] += (float)(duration.count() / 1000);

		// Opportunistically pre-fetch next batch onto the alternate buffer.
		// memcpy_input uses cudaMemcpyAsync on procs[nxt]->CUDA_stream, so the
		// H2D for batch N+1 overlaps with GPU kernels for batch N below.
		// Skip when proc0 == proc1 (single-buffer fallback: no second independent buffer).
		int nxt = 1 - cur;
		parsed_chunk_t *pc_nxt;
		if (procs[0] != procs[1] && dispatch_queue.try_dequeue(pc_nxt)) {
			dispatch_queue_size--;
			ac_pre = new aligned_chunk_t;
			ac_pre->chunk_offset = pc_nxt->chunk_offset;
			ac_pre->chunk_size   = pc_nxt->chunk_size;
			ac_pre->seq_offsets  = pc_nxt->seq_offsets;
			ac_pre->seq          = pc_nxt->seq;
			ac_pre->name_offsets = pc_nxt->name_offsets;
			ac_pre->name         = pc_nxt->name;
			ac_pre->qual_offsets = pc_nxt->qual_offsets;
			ac_pre->qual         = pc_nxt->qual;
			delete pc_nxt;
			dispatched_cnt += ac_pre->chunk_size;
			// set n_processed for the prefetched batch before its H2D starts.
			// The prefetched batch will be processed AFTER the current one, so its
			// batch_offset = reads completed before current batch + current batch size.
			procs[nxt]->n_processed = n_reads_completed + (int64_t)ac->chunk_size;
			memcpy_input(ac_pre->chunk_size, procs[nxt],
					ac_pre->seq.data(), ac_pre->seq_offsets.data());
			prefetch_valid = true;
		}

		// set correct batch_offset (n_processed) for sort_regions hash tiebreaker.
		procs[cur]->n_processed = n_reads_completed;

		// launch compute kernels.
		TIMER_START();
		if(bwa_align(gpuid, procs[cur], aux->g3_opt) != 0){
		}
		TIMER_END(0, "computed a chunk");
		n_reads_completed += (int64_t)ac->chunk_size;  // advance after batch completes
		tprof[gpuid][COMPUTE_TOTAL] += ((float)duration.count() / 1000);


		// allocate per-batch event (owned by task; worker_samgen destroys it).
		int d2h_slot = cur;
		void *d2h_done = cuda_d2h_event_create();
		TIMER_START();
		memcpy_output_async(ac, procs[d2h_slot], o_stream, d2h_stream);
		cuda_d2h_event_record(d2h_done, d2h_stream);
		TIMER_END(0, "async D2H issued");
		tprof[gpuid][PULL_TOTAL] += ((float)duration.count() / 1000);

		// Rotate to the pre-fetched buffer for the next iteration.
		if (prefetch_valid) cur = nxt;

		// post to samgen; worker_samgen syncs d2h_done and then destroys the event.
		{
			std::lock_guard<std::mutex> lk(samgen_qs[gpuid].mu);
			samgen_qs[gpuid].q.push_back({ac, procs[d2h_slot], d2h_done});
			samgen_qs[gpuid].cv.notify_one();
		}
	}

	cuda_o_stream_destroy(o_stream);
	cuda_d2h_stream_destroy(d2h_stream);

	active_dispatcher_cnt--;
	LARGE_TIMER_END(0, "dispatched chunks");
	tprof[gpuid][ALIGNER_TOP] += (float)(large_duration.count() / 1000);
	return;
}

// background SAM-gen worker — one thread per GPU.
// Waits for async D2H to complete, formats SAM using pinned h_out_* arrays,
// pushes result to write_queue.
void worker_samgen(
		int gpuid,
		pipeline_aux_t *aux,
		std::priority_queue<
		aligned_chunk_t *, std::vector<aligned_chunk_t *>, writequeue_compare
		> &write_queue)
{
	TIMER_INIT();
	cuda_set_device(gpuid);
	while (true) {
		SamGenTask task;
		{
			std::unique_lock<std::mutex> lk(samgen_qs[gpuid].mu);
			samgen_qs[gpuid].cv.wait(lk, [&]{ return !samgen_qs[gpuid].q.empty(); });
			task = samgen_qs[gpuid].q.front();
			samgen_qs[gpuid].q.pop_front();
		}
		if (!task.ac) break; // nullptr sentinel = shutdown

		// Wait for async D2H; then release the event (task owns it).
		cuda_d2h_event_sync(task.d2h_done);
		cuda_d2h_event_destroy(task.d2h_done);

		aligned_chunk_t   *ac   = task.ac;
		process_data_t    *proc = task.proc;

		TIMER_START();
		for (int seq_id = 0; seq_id < ac->chunk_size; seq_id++) {
			uint8_t *seqbuf;
			int name_len, qual_len, seq_len;
			name_len = ac->name_offsets.at(seq_id + 1) - ac->name_offsets.at(seq_id);
			qual_len = ac->qual_offsets.at(seq_id + 1) - ac->qual_offsets.at(seq_id);
			seqbuf   = ac->seq.data() + ac->seq_offsets.at(seq_id);
			seq_len  = ac->seq_offsets.at(seq_id + 1) - ac->seq_offsets.at(seq_id);

			int offset      = ac->offsets.at(seq_id);
			int offset_next = ac->offsets.at(seq_id + 1);
			if (offset_next <= offset) {
				ac->sambuf += ac->name.substr(ac->name_offsets.at(seq_id), name_len);
				ac->sambuf += "\t4\t*\t0\t0\t*\t*\t0\t0\t";
				uint8_t *p = seqbuf;
				while (p < seqbuf + seq_len)
					ac->sambuf += "ACGTN"[(int)*p++];
				ac->sambuf += "\t";
				ac->sambuf += ac->qual.substr(ac->qual_offsets.at(seq_id), qual_len);
				ac->sambuf += "\n";
				continue;
			}
			for (int aln_id = offset; aln_id < offset_next; aln_id++) {
				// read alignment data from pinned h_out_* slot arrays
				// (written by async D2H; event ensures data is ready).
				int rid  = proc->h_out_rids[aln_id];
				if (rid < 0) continue;
				int flag = proc->h_out_flags[aln_id];
				int mapq = proc->h_out_mapqs[aln_id];

				ac->sambuf += ac->name.substr(ac->name_offsets.at(seq_id), name_len);
				ac->sambuf += "\t";
				ac->sambuf += std::to_string(flag);
				ac->sambuf += "\t";
				ac->sambuf += aux->idx->bns->anns[rid].name;
				ac->sambuf += "\t";
				ac->sambuf += std::to_string(proc->h_out_positions[aln_id] + 1);
				ac->sambuf += "\t";
				ac->sambuf += std::to_string(mapq);
				ac->sambuf += "\t";

				int ncigars       = proc->h_out_ncigars[aln_id];
				uint32_t *cigars  = proc->h_out_cigars + aln_id * MAX_N_CIGAR;
				int hclip_front = 0, hclip_back = 0;
				if ((flag & 0x800) && ncigars > 0) {
					if ((cigars[0] & 0xf) == 3) {
						hclip_front = BAM2LEN(cigars[0]);
						cigars[0]   = (hclip_front << 4) | 4;
					}
					if (ncigars > 1 && (cigars[ncigars - 1] & 0xf) == 3) {
						hclip_back         = BAM2LEN(cigars[ncigars - 1]);
						cigars[ncigars - 1] = (hclip_back << 4) | 4;
					}
				}
				if (ncigars == 0) {
					ac->sambuf += "*";
				} else {
					uint32_t *cp = cigars;
					int nc = ncigars;
					while (nc-- > 0) {
						ac->sambuf += std::to_string(BAM2LEN(*cp));
						ac->sambuf += BAM2OP(*cp++);
					}
				}
				ac->sambuf += "\t*\t0\t0\t";

				int seq_start = 0, seq_end = seq_len;
				if (flag & 0x800) {
					if (flag & 0x10) { seq_start += hclip_back;  seq_end -= hclip_front; }
					else             { seq_start += hclip_front; seq_end -= hclip_back;  }
				}
				if (flag & 0x10) {
					static const char comp_table[] = "TGCAN";
					for (int i = seq_end - 1; i >= seq_start; i--)
						ac->sambuf += comp_table[(int)seqbuf[i]];
				} else {
					uint8_t *p = seqbuf + seq_start;
					while (p < seqbuf + seq_end)
						ac->sambuf += "ACGTN"[(int)*p++];
				}
				ac->sambuf += "\t";

				int qual_start = (flag & 0x800) ? ((flag & 0x10) ? hclip_back  : hclip_front) : 0;
				int qual_end   = qual_len - ((flag & 0x800) ? ((flag & 0x10) ? hclip_front : hclip_back) : 0);
				if (flag & 0x10) {
					std::string q = ac->qual.substr(ac->qual_offsets.at(seq_id) + qual_start, qual_end - qual_start);
					std::reverse(q.begin(), q.end());
					ac->sambuf += q;
				} else {
					ac->sambuf += ac->qual.substr(ac->qual_offsets.at(seq_id) + qual_start, qual_end - qual_start);
				}
				ac->sambuf += "\n";
			}
		}
		TIMER_END(0, "samgen a chunk");
		tprof[gpuid][SAMGEN_TOTAL] += (float)(duration.count() / 1000);

		{
			std::lock_guard<std::mutex> lock(write_queue_mutex);
			write_queue.push(ac);
		}
	}
	active_samgen_cnt--;
}

void worker_write(
		std::priority_queue<
		aligned_chunk_t *, std::vector<aligned_chunk_t *>, writequeue_compare
		> &write_queue,
		std::ostream *samout,
		g3_opt_t *g3_opt
		)
{
	TIMER_INIT();
	float write_time = 0;
	aligned_chunk_t *ac;
	bool first = true;
	while(true) {
		bool all_done = (active_samgen_cnt == 0);
		ac = nullptr;
		{
			std::lock_guard<std::mutex> lock(write_queue_mutex);
			if (!write_queue.empty()) {
				ac = write_queue.top();
				write_queue.pop();
			}
		}
		if (!ac) {
			// Queue empty: exit only once all samgen threads have finished.
			if (all_done) break;
			continue;
		}
		if(first){
			std::cerr << "* WRITING OUTPUT STARTS...\n";
			first = false;
		}
		TIMER_START();
		*samout << ac->sambuf;
		samout->flush();
		TIMER_END(0, "");
		write_time += (float)(duration.count() / 1000);
		written_cnt += ac->chunk_size;
		delete ac;
		std::cerr << "* wrote sams for " << written_cnt << " reads\n";
	}

	for(int gpuid = 0; gpuid < g3_opt->num_use_gpus; gpuid++){
		tprof[gpuid][FILE_OUTPUT] = write_time;
	}
}

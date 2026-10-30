//! Speed Demon 2026 - Track B (Index)
//! High-throughput boolean & phrase search over text corpus.
//!
//! Architectural highlights:
//! 1. Query-first vocabulary pruning: extracts query terms upfront.
//! 2. Postings lists with sorted doc IDs: two-pointer linear scans for AND/OR/NOT.
//! 3. Candidate-filtered phrase evaluation: intersects term postings to narrow candidates,
//!    then verifies consecutive token matching on the fly without positional indexing.
//! 4. 12-thread parallel document parsing and parallel query execution.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::thread;

#[inline(always)]
fn is_token_byte(b: u8) -> bool {
    (b >= b'a' && b <= b'z') || (b >= b'0' && b <= b'9') || (b >= b'A' && b <= b'Z')
}

#[inline(always)]
fn to_lower_ascii(b: u8) -> u8 {
    if b >= b'A' && b <= b'Z' {
        b + 32
    } else {
        b
    }
}

/// Fast open-addressing hash map from &[u8] to u32
struct TermMap {
    keys: Vec<Vec<u8>>,
    slots: Vec<(u64, u32)>,
    mask: usize,
}

impl TermMap {
    fn with_capacity(cap: usize) -> Self {
        let size = (cap * 2).next_power_of_two().max(262144);
        Self {
            keys: Vec::with_capacity(cap),
            slots: vec![(0, u32::MAX); size],
            mask: size - 1,
        }
    }

    #[inline(always)]
    fn hash(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
        h
    }

    fn insert(&mut self, key: &[u8]) -> u32 {
        let h = Self::hash(key);
        let mut idx = (h as usize) & self.mask;
        loop {
            let (entry_h, id) = self.slots[idx];
            if id == u32::MAX {
                let new_id = self.keys.len() as u32;
                self.keys.push(key.to_vec());
                self.slots[idx] = (h, new_id);
                return new_id;
            }
            if entry_h == h && self.keys[id as usize] == key {
                return id;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn get(&self, key: &[u8]) -> Option<u32> {
        let h = Self::hash(key);
        let mut idx = (h as usize) & self.mask;
        loop {
            let (entry_h, id) = self.slots[idx];
            if id == u32::MAX {
                return None;
            }
            if entry_h == h && self.keys[id as usize] == key {
                return Some(id);
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

enum Query {
    Bool {
        terms: Vec<Option<u32>>,
        ops: Vec<u8>, // 0 = AND, 1 = OR, 2 = NOT
    },
    Phrase {
        terms: Vec<Option<u32>>,
    },
}

#[inline(always)]
fn intersect(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            out.push(a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
}

#[inline(always)]
fn union(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            out.push(a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

#[inline(always)]
fn difference(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
}

#[inline(always)]
fn write_u64(buf: &mut Vec<u8>, mut n: u64) {
    if n == 0 {
        buf.push(b'0');
        return;
    }
    let mut temp = [0u8; 20];
    let mut i = 0;
    while n > 0 {
        temp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        buf.push(temp[i]);
    }
}

#[inline(always)]
fn write_u32(buf: &mut Vec<u8>, mut n: u32) {
    if n == 0 {
        buf.push(b'0');
        return;
    }
    let mut temp = [0u8; 10];
    let mut i = 0;
    while n > 0 {
        temp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        buf.push(temp[i]);
    }
}

#[cfg(target_os = "linux")]
struct MmapBuf {
    ptr: *const u8,
    len: usize,
}

#[cfg(target_os = "linux")]
unsafe impl Send for MmapBuf {}
#[cfg(target_os = "linux")]
unsafe impl Sync for MmapBuf {}

#[cfg(target_os = "linux")]
impl std::ops::Deref for MmapBuf {
    type Target = [u8];
    #[inline(always)]
    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

#[cfg(target_os = "linux")]
fn map_file(path: &std::path::Path) -> Result<MmapBuf, std::io::Error> {
    use std::os::unix::io::AsRawFd;
    let f = File::open(path)?;
    let len = f.metadata()?.len() as usize;
    if len == 0 {
        return Ok(MmapBuf {
            ptr: std::ptr::null(),
            len: 0,
        });
    }
    let fd = f.as_raw_fd();
    let mut addr: *mut u8;
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") 9usize,
            in("rdi") 0usize,
            in("rsi") len,
            in("rdx") 1usize,
            in("r10") 2usize,
            in("r8") fd as usize,
            in("r9") 0usize,
            lateout("rax") addr,
            lateout("rcx") _,
            lateout("r11") _,
        );
    }
    if (addr as isize) < 0 && (addr as isize) >= -4095 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(MmapBuf { ptr: addr, len })
}

#[cfg(not(target_os = "linux"))]
fn map_file(path: &std::path::Path) -> Result<Vec<u8>, std::io::Error> {
    std::fs::read(path)
}

fn main() {
    let mut data_dir = PathBuf::from("/data");
    let mut out_dir = PathBuf::from("/out");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--data" && i + 1 < args.len() {
            data_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else if args[i] == "--out" && i + 1 < args.len() {
            out_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    std::fs::create_dir_all(&out_dir).ok();
    let output_path = out_dir.join("results.txt");

    let queries_path = data_dir.join("queries.txt");
    let docs_path = data_dir.join("docs.txt");

    let queries_buf = match std::fs::read(&queries_path) {
        Ok(b) => b,
        Err(_) => {
            File::create(&output_path).ok();
            return;
        }
    };

    let docs_buf = match map_file(&docs_path) {
        Ok(b) => b,
        Err(_) => {
            File::create(&output_path).ok();
            return;
        }
    };

    // 1. Parse queries and build query term map
    let mut term_map = TermMap::with_capacity(131072);
    let mut parsed_queries: Vec<Query> = Vec::with_capacity(50000);

    let mut q_ptr = 0;
    let q_len = queries_buf.len();
    while q_ptr < q_len {
        let line_start = q_ptr;
        while q_ptr < q_len && queries_buf[q_ptr] != b'\n' {
            q_ptr += 1;
        }
        let mut line_end = q_ptr;
        if line_end > line_start && queries_buf[line_end - 1] == b'\r' {
            line_end -= 1;
        }
        if q_ptr < q_len {
            q_ptr += 1;
        }

        let line = &queries_buf[line_start..line_end];
        if line.starts_with(b"BOOL ") {
            let mut parts = line[5..].split(|&b| b == b' ');
            let first_term = parts.next().unwrap();
            let mut terms = vec![Some(term_map.insert(first_term))];
            let mut ops = Vec::new();

            while let Some(op_bytes) = parts.next() {
                let op = match op_bytes {
                    b"AND" => 0,
                    b"OR" => 1,
                    b"NOT" => 2,
                    _ => 0,
                };
                ops.push(op);
                let term = parts.next().unwrap();
                terms.push(Some(term_map.insert(term)));
            }
            parsed_queries.push(Query::Bool { terms, ops });
        } else if line.starts_with(b"PHRASE \"") {
            let inner = &line[8..line.len() - 1];
            let mut terms = Vec::new();
            for t in inner.split(|&b| b == b' ') {
                terms.push(Some(term_map.insert(t)));
            }
            parsed_queries.push(Query::Phrase { terms });
        }
    }

    let num_query_terms = term_map.keys.len();

    // 2. Scan docs.txt to index document boundaries and populate posting lists
    let d_len = docs_buf.len();
    let mut doc_text_ranges: Vec<(u32, u32)> = Vec::with_capacity(250000);

    let mut ptr = 0;
    while ptr < d_len {
        let tab_pos = match docs_buf[ptr..].iter().position(|&b| b == b'\t') {
            Some(pos) => ptr + pos,
            None => break,
        };
        let text_start = (tab_pos + 1) as u32;

        let nl_pos = match docs_buf[tab_pos + 1..].iter().position(|&b| b == b'\n') {
            Some(pos) => tab_pos + 1 + pos,
            None => d_len,
        };
        let mut text_end = nl_pos as u32;
        if text_end > text_start && docs_buf[(text_end - 1) as usize] == b'\r' {
            text_end -= 1;
        }

        doc_text_ranges.push((text_start, text_end));
        ptr = if nl_pos < d_len { nl_pos + 1 } else { d_len };
    }

    let num_docs = doc_text_ranges.len();
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(12);

    let chunk_size = (num_docs + num_threads - 1) / num_threads;
    let doc_chunks: Vec<(usize, usize)> = (0..num_threads)
        .map(|i| {
            let s = (i * chunk_size).min(num_docs);
            let e = ((i + 1) * chunk_size).min(num_docs);
            (s, e)
        })
        .collect();

    struct ThreadIndexResult {
        records: Vec<(u32, u32)>, // (tid, doc_id)
        counts: Vec<u32>,         // size num_query_terms
    }

    // Each thread extracts tokens for its documents and populates thread-local records
    let thread_indexes: Vec<ThreadIndexResult> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for &(d_start, d_end) in &doc_chunks {
            let ranges = &doc_text_ranges[d_start..d_end];
            let buf = &docs_buf;
            let tm = &term_map;

            handles.push(s.spawn(move || {
                let mut records = Vec::with_capacity(ranges.len() * 20);
                let mut counts = vec![0u32; num_query_terms];
                let mut seen_stamp = vec![0u32; num_query_terms];
                let mut token_buf = [0u8; 32];

                for (idx, &(t_start, t_end)) in ranges.iter().enumerate() {
                    let doc_id = (d_start + idx) as u32;
                    let current_stamp = doc_id + 1;

                    let slice = &buf[t_start as usize..t_end as usize];
                    let mut pos = 0;
                    let slen = slice.len();

                    while pos < slen {
                        while pos < slen && !is_token_byte(slice[pos]) {
                            pos += 1;
                        }
                        let mut tok_len = 0;
                        while pos < slen && is_token_byte(slice[pos]) {
                            if tok_len < 32 {
                                token_buf[tok_len] = to_lower_ascii(slice[pos]);
                                tok_len += 1;
                            }
                            pos += 1;
                        }
                        if tok_len > 0 {
                            if let Some(tid) = tm.get(&token_buf[..tok_len]) {
                                let tid_idx = tid as usize;
                                if seen_stamp[tid_idx] != current_stamp {
                                    seen_stamp[tid_idx] = current_stamp;
                                    records.push((tid, doc_id));
                                    counts[tid_idx] += 1;
                                }
                            }
                        }
                    }
                }
                ThreadIndexResult { records, counts }
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Compute prefix sums over global query terms
    let mut offsets = vec![0u32; num_query_terms + 1];
    let mut thread_term_offsets: Vec<Vec<u32>> = vec![vec![0u32; num_query_terms]; num_threads];

    let mut acc = 0u32;
    for tid in 0..num_query_terms {
        offsets[tid] = acc;
        let mut cur = acc;
        for t in 0..num_threads {
            thread_term_offsets[t][tid] = cur;
            cur += thread_indexes[t].counts[tid];
        }
        acc = cur;
    }
    offsets[num_query_terms] = acc;
    let total_postings = acc as usize;

    let mut doc_ids = vec![0u32; total_postings];
    let doc_ptr = doc_ids.as_mut_ptr() as usize;

    // Parallel scatter into contiguous doc_ids
    thread::scope(|s| {
        for (t_idx, tr) in thread_indexes.into_iter().enumerate() {
            let mut t_offs = std::mem::take(&mut thread_term_offsets[t_idx]);
            s.spawn(move || {
                let raw_ptr = doc_ptr as *mut u32;
                for &(tid, doc_id) in &tr.records {
                    let pos = t_offs[tid as usize] as usize;
                    t_offs[tid as usize] += 1;
                    unsafe {
                        *raw_ptr.add(pos) = doc_id;
                    }
                }
            });
        }
    });

    struct CsrPostings<'a> {
        offsets: &'a [u32],
        doc_ids: &'a [u32],
    }

    impl<'a> CsrPostings<'a> {
        #[inline(always)]
        fn get(&self, tid: u32) -> &'a [u32] {
            let s = self.offsets[tid as usize] as usize;
            let e = self.offsets[tid as usize + 1] as usize;
            &self.doc_ids[s..e]
        }
    }

    let postings = CsrPostings {
        offsets: &offsets,
        doc_ids: &doc_ids,
    };

    // 3. Parallel query execution
    let queries_per_thread = (parsed_queries.len() + num_threads - 1) / num_threads;
    let query_chunks: Vec<(usize, usize)> = (0..num_threads)
        .map(|i| {
            let s = (i * queries_per_thread).min(parsed_queries.len());
            let e = ((i + 1) * queries_per_thread).min(parsed_queries.len());
            (s, e)
        })
        .collect();

    let output_chunks: Vec<Vec<u8>> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for &(q_start, q_end) in &query_chunks {
            let queries = &parsed_queries[q_start..q_end];
            let post_lists = &postings;
            let ranges = &doc_text_ranges;
            let d_buf = &docs_buf;
            let tm = &term_map;

            handles.push(s.spawn(move || {
                let mut out_bytes = Vec::with_capacity(512 * 1024);
                let mut r_set = Vec::new();
                let mut temp_buf = Vec::new();
                let mut cand_docs = Vec::new();
                let mut token_buf = [0u8; 32];
                let mut doc_tokens = Vec::with_capacity(256);

                for q in queries {
                    match q {
                        Query::Bool { terms, ops } => {
                            if let Some(first_tid) = terms[0] {
                                r_set.clear();
                                r_set.extend_from_slice(post_lists.get(first_tid));
                            } else {
                                r_set.clear();
                            }

                            for (op_idx, &op) in ops.iter().enumerate() {
                                let term_tid = terms[op_idx + 1];
                                match op {
                                    0 => {
                                        // AND
                                        if let Some(tid) = term_tid {
                                            intersect(
                                                &r_set,
                                                post_lists.get(tid),
                                                &mut temp_buf,
                                            );
                                            std::mem::swap(&mut r_set, &mut temp_buf);
                                        } else {
                                            r_set.clear();
                                        }
                                    }
                                    1 => {
                                        // OR
                                        if let Some(tid) = term_tid {
                                            union(&r_set, post_lists.get(tid), &mut temp_buf);
                                            std::mem::swap(&mut r_set, &mut temp_buf);
                                        }
                                    }
                                    2 => {
                                        // NOT
                                        if let Some(tid) = term_tid {
                                            difference(
                                                &r_set,
                                                post_lists.get(tid),
                                                &mut temp_buf,
                                            );
                                            std::mem::swap(&mut r_set, &mut temp_buf);
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            if r_set.is_empty() {
                                out_bytes.extend_from_slice(b"0 0\n");
                            } else {
                                let count = r_set.len() as u64;
                                let sum: u64 = r_set.iter().map(|&x| x as u64).sum();
                                write_u64(&mut out_bytes, count);
                                out_bytes.push(b' ');
                                write_u64(&mut out_bytes, sum);
                                let take_m = r_set.len().min(10);
                                for k in 0..take_m {
                                    out_bytes.push(b' ');
                                    write_u32(&mut out_bytes, r_set[k]);
                                }
                                out_bytes.push(b'\n');
                            }
                        }
                        Query::Phrase { terms } => {
                            let mut all_present = true;
                            for t in terms {
                                if t.is_none() {
                                    all_present = false;
                                    break;
                                }
                            }

                            if !all_present {
                                out_bytes.extend_from_slice(b"0 0\n");
                                continue;
                            }

                            // Intersect postings of phrase terms to find candidate documents
                            let first_tid = terms[0].unwrap();
                            cand_docs.clear();
                            cand_docs.extend_from_slice(post_lists.get(first_tid));

                            for &t in &terms[1..] {
                                let tid = t.unwrap();
                                intersect(&cand_docs, post_lists.get(tid), &mut temp_buf);
                                std::mem::swap(&mut cand_docs, &mut temp_buf);
                                if cand_docs.is_empty() {
                                    break;
                                }
                            }

                            if cand_docs.is_empty() {
                                out_bytes.extend_from_slice(b"0 0\n");
                                continue;
                            }

                            // Verify phrase in candidate documents
                            let k = terms.len();
                            let phrase_tids: Vec<u32> = terms.iter().map(|t| t.unwrap()).collect();

                            let mut count = 0u64;
                            let mut sum = 0u64;
                            let mut smallest = [0u32; 10];
                            let mut found_count = 0;

                            for &doc_id in &cand_docs {
                                let (t_start, t_end) = ranges[doc_id as usize];
                                let slice = &d_buf[t_start as usize..t_end as usize];

                                doc_tokens.clear();
                                let mut pos = 0;
                                let slen = slice.len();
                                while pos < slen {
                                    while pos < slen && !is_token_byte(slice[pos]) {
                                        pos += 1;
                                    }
                                    let mut tok_len = 0;
                                    while pos < slen && is_token_byte(slice[pos]) {
                                        if tok_len < 32 {
                                            token_buf[tok_len] = to_lower_ascii(slice[pos]);
                                            tok_len += 1;
                                        }
                                        pos += 1;
                                    }
                                    if tok_len > 0 {
                                        let tid = tm.get(&token_buf[..tok_len]).unwrap_or(u32::MAX);
                                        doc_tokens.push(tid);
                                    }
                                }

                                let mut matched = false;
                                if doc_tokens.len() >= k {
                                    for window in doc_tokens.windows(k) {
                                        if window == phrase_tids.as_slice() {
                                            matched = true;
                                            break;
                                        }
                                    }
                                }

                                if matched {
                                    count += 1;
                                    sum += doc_id as u64;
                                    if found_count < 10 {
                                        smallest[found_count] = doc_id;
                                        found_count += 1;
                                    }
                                }
                            }

                            if count == 0 {
                                out_bytes.extend_from_slice(b"0 0\n");
                            } else {
                                write_u64(&mut out_bytes, count);
                                out_bytes.push(b' ');
                                write_u64(&mut out_bytes, sum);
                                for m in 0..found_count {
                                    out_bytes.push(b' ');
                                    write_u32(&mut out_bytes, smallest[m]);
                                }
                                out_bytes.push(b'\n');
                            }
                        }
                    }
                }
                out_bytes
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out_file = File::create(&output_path).expect("failed to create results.txt");
    for chunk in output_chunks {
        out_file
            .write_all(&chunk)
            .expect("failed to write results.txt");
    }
}

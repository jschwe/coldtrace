#include <coldtrace/config.h>
#include <coldtrace/libc.h>
#include <coldtrace/version.h>
#include <coldtrace/writer.h>
#include <dice/compiler.h>
#include <dice/log.h>
#include <dice/mempool.h>
#include <dice/self.h>
#include <dice/types.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vsync/atomic.h>
#include <vsync/spinlock/caslock.h>

#if defined(__linux__)
    #include <linux/futex.h>
    #include <sys/syscall.h>
#endif

#define FORMAT_EXPANSION_SPACE 20
#define FILE_NAME_SIZE         160
#define FILE_PERMISSIONS                                                       \
    (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)

enum flush_state {
    FLUSH_IDLE = 0,
    FLUSH_QUEUED,
    FLUSH_TAKEN,
};

/* A filled fragment that has to be written to its file. The thread that
 * moves the state from QUEUED to TAKEN writes it and sets it back to IDLE. */
struct flush_job {
    vatomic32_t state;
    int error;
    pid_t pid;
    const void *data;
    size_t len;
    char file_name[FILE_NAME_SIZE];
};

struct writer_impl;

/* Fragment buffers of a writer: one is filled while the other is flushed.
 * These are recycled through a free list and never released, so a stale
 * queue entry always points to a valid job. The lock keeps the exit flush
 * away from a rotation in progress. */
struct writer_buffers {
    uint64_t *buf[2];
    struct flush_job job[2];
    unsigned cur;
    caslock_t lock;
    struct writer_impl *owner;
    pid_t pid;
    struct writer_buffers *next_live;
    struct writer_buffers *prev_live;
    struct writer_buffers *next_free;
};

struct writer_impl {
    bool initd;
    bool failed;
    uint64_t *buffer;
    uint64_t size;
    uint64_t offset;
    uint64_t tid;
    uint32_t enumerator;
    metadata_t *md;
    struct writer_buffers *bufs;
};

// Ensure the size of implementation matches the public size.
STATIC_ASSERT(sizeof(struct writer_impl) == sizeof(struct coldtrace_writer),
              "incorrect writer_impl size");

static bool
write_all_(int fd, const void *buf, size_t count)
{
    const char *p    = (const char *)buf;
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t w = write(fd, p, remaining);
        if (w == -1) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        p += (size_t)w;
        remaining -= (size_t)w;
    }
    return true;
}

// -----------------------------------------------------------------------------
// background flusher
// -----------------------------------------------------------------------------

#define FLUSH_QUEUE_SIZE 64

static struct {
    caslock_t lock;
    struct flush_job *queue[FLUSH_QUEUE_SIZE];
    uint32_t head;
    uint32_t tail;
    struct flush_job *current; // job dequeued by the flusher thread
    pid_t pid;                 // process running the flusher, -pid if failed
    vatomic32_t seq;           // bumped on enqueue, the flusher waits on it
    vatomic32_t done;          // bumped after each job, drain waits on it
    struct writer_buffers *free_bufs;
    struct writer_buffers *live_bufs;
    size_t nlive;
} flusher_;

#if defined(__linux__)
static void
futex_wait_(vatomic32_t *addr, uint32_t val)
{
    syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
}

static void
futex_wake_(vatomic32_t *addr)
{
    syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL, NULL, 0);
}
#else
static void
futex_wait_(vatomic32_t *addr, uint32_t val)
{
    if (vatomic32_read_acq(addr) == val) {
        usleep(100);
    }
}

static void
futex_wake_(vatomic32_t *addr)
{
    (void)addr;
}
#endif

// Only calls functions that dice does not intercept, so that the flusher
// thread stays invisible to dice.
static void
flush_job_write_(struct flush_job *job)
{
    job->error = 0;
    int fd =
        open(job->file_name, O_WRONLY | O_CREAT | O_TRUNC, FILE_PERMISSIONS);
    if (fd == -1) {
        job->error = errno;
        return;
    }
    if (!write_all_(fd, job->data, job->len)) {
        job->error = errno;
    }
    close(fd);
}

static struct flush_job *
flusher_dequeue_(void)
{
    struct flush_job *job = NULL;
    if (flusher_.head != flusher_.tail) {
        job = flusher_.queue[flusher_.head++ % FLUSH_QUEUE_SIZE];
    }
    return job;
}

static void *
flusher_main_(void *arg)
{
    (void)arg;
    for (;;) {
        uint32_t seq = vatomic32_read_acq(&flusher_.seq);
        for (;;) {
            caslock_acquire(&flusher_.lock);
            struct flush_job *job = flusher_dequeue_();
            flusher_.current      = job;
            caslock_release(&flusher_.lock);
            if (job == NULL) {
                break;
            }
            if (vatomic32_cmpxchg_acq(&job->state, FLUSH_QUEUED, FLUSH_TAKEN) ==
                FLUSH_QUEUED) {
                flush_job_write_(job);
                vatomic32_write_rel(&job->state, FLUSH_IDLE);
                futex_wake_(&job->state);
            }
            caslock_acquire(&flusher_.lock);
            flusher_.current = NULL;
            caslock_release(&flusher_.lock);
            vatomic32_inc_rel(&flusher_.done);
            futex_wake_(&flusher_.done);
        }
        futex_wait_(&flusher_.seq, seq);
    }
    return NULL;
}

// A fork child must not inherit the queue lock held by another thread.
static void
flusher_atfork_prepare_(void)
{
    caslock_acquire(&flusher_.lock);
}

static void
flusher_atfork_release_(void)
{
    caslock_release(&flusher_.lock);
}

// Starts the flusher thread for this process if needed. Must hold the lock.
static bool
flusher_start_(pid_t pid)
{
    if (flusher_.pid == pid) {
        return true;
    }
    if (flusher_.pid == -pid) {
        return false;
    }
    // A different pid means we are in a fork child: the queued jobs belong
    // to the parent.
    flusher_.head    = 0;
    flusher_.tail    = 0;
    flusher_.current = NULL;

    // Bypass the dice interposer so the thread is not traced, and keep
    // signals of the traced program away from it.
    typedef int (*create_f)(pthread_t *, const pthread_attr_t *,
                            void *(*)(void *), void *);
    create_f create = (create_f)COLDTRACE_REAL_("pthread_create");
    if (create == NULL) {
        flusher_.pid = -pid;
        return false;
    }
    sigset_t all, old;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &old);
    pthread_t thread;
    int err = create(&thread, NULL, flusher_main_, NULL);
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    if (err != 0) {
        log_warn("could not start flusher thread: %s", strerror(err));
        flusher_.pid = -pid;
        return false;
    }
    pthread_detach(thread);
    static bool atfork_ = false;
    if (!atfork_) {
        atfork_ =
            pthread_atfork(flusher_atfork_prepare_, flusher_atfork_release_,
                           flusher_atfork_release_) == 0;
    }
    flusher_.pid = pid;
    return true;
}

static bool
flusher_enqueue_(struct flush_job *job)
{
    bool queued = false;
    caslock_acquire(&flusher_.lock);
    if (flusher_start_(job->pid) &&
        flusher_.tail - flusher_.head < FLUSH_QUEUE_SIZE) {
        vatomic32_write_rel(&job->state, FLUSH_QUEUED);
        flusher_.queue[flusher_.tail++ % FLUSH_QUEUE_SIZE] = job;
        queued                                             = true;
    }
    caslock_release(&flusher_.lock);
    if (queued) {
        vatomic32_inc_rel(&flusher_.seq);
        futex_wake_(&flusher_.seq);
    }
    return queued;
}

// Waits until the job's buffer can be reused, writing the job itself if the
// flusher has not picked it up yet. Returns false if the write failed.
static bool
flush_job_wait_(struct flush_job *job)
{
    for (;;) {
        uint32_t state = vatomic32_read_acq(&job->state);
        if (state == FLUSH_IDLE) {
            break;
        }
        if (job->pid != getpid()) {
            // queued before a fork, the parent writes it
            vatomic32_write_rlx(&job->state, FLUSH_IDLE);
            break;
        }
        if (state == FLUSH_QUEUED) {
            if (vatomic32_cmpxchg_acq(&job->state, FLUSH_QUEUED, FLUSH_TAKEN) ==
                FLUSH_QUEUED) {
                flush_job_write_(job);
                vatomic32_write_rel(&job->state, FLUSH_IDLE);
                break;
            }
            continue;
        }
        futex_wait_(&job->state, FLUSH_TAKEN);
    }
    if (job->error != 0) {
        log_warn("write flush %s: %s", job->file_name, strerror(job->error));
        job->error = 0;
        return false;
    }
    return true;
}

// Writes the partial fragments of threads that are still running, which would
// otherwise be lost at exit.
static void
flush_live_writers_(void)
{
    caslock_acquire(&flusher_.lock);
    size_t n = flusher_.nlive;
    struct writer_buffers *live[n > 0 ? n : 1];
    size_t i = 0;
    for (struct writer_buffers *b = flusher_.live_bufs; b && i < n;
         b                        = b->next_live) {
        live[i++] = b;
    }
    caslock_release(&flusher_.lock);

    pid_t pid = getpid();
    for (size_t k = 0; k < i; k++) {
        struct writer_buffers *b = live[k];
        // after a fork, the inherited writers and their locks belong to the
        // parent's threads
        if (b->pid != pid) {
            continue;
        }
        caslock_acquire(&b->lock);
        struct writer_impl *impl = b->owner;
        if (impl != NULL && impl->buffer != NULL) {
            struct flush_job job = {.data = impl->buffer, .len = impl->offset};
            snprintf(job.file_name, sizeof(job.file_name),
                     coldtrace_get_file_pattern(), impl->tid, impl->enumerator);
            flush_job_write_(&job);
        }
        caslock_release(&b->lock);
    }
}

// Writes what is still queued when the process exits, then the fragments of
// threads that are still running.
static DICE_DTOR void
flusher_drain_(void)
{
    // in a fork child the queue holds the parent's jobs
    while (flusher_.pid == getpid()) {
        caslock_acquire(&flusher_.lock);
        struct flush_job *job = flusher_dequeue_();
        bool busy             = flusher_.current != NULL;
        uint32_t done         = vatomic32_read_rlx(&flusher_.done);
        caslock_release(&flusher_.lock);
        if (job != NULL) {
            if (vatomic32_cmpxchg_acq(&job->state, FLUSH_QUEUED, FLUSH_TAKEN) ==
                FLUSH_QUEUED) {
                flush_job_write_(job);
                vatomic32_write_rel(&job->state, FLUSH_IDLE);
                futex_wake_(&job->state);
            }
            continue;
        }
        if (!busy) {
            break;
        }
        futex_wait_(&flusher_.done, done);
    }
    flush_live_writers_();
}

static struct writer_buffers *
buffers_get_(struct writer_impl *owner)
{
    caslock_acquire(&flusher_.lock);
    struct writer_buffers *b = flusher_.free_bufs;
    if (b != NULL) {
        flusher_.free_bufs = b->next_free;
    }
    caslock_release(&flusher_.lock);
    if (b == NULL) {
        b = mempool_alloc(sizeof(*b));
        if (b == NULL) {
            return NULL;
        }
        coldtrace_memset(b, 0, sizeof(*b));
    }
    b->buf[0] = b->buf[1] = NULL;
    b->cur                = 0;

    caslock_acquire(&b->lock);
    b->owner = owner;
    b->pid   = getpid();
    caslock_release(&b->lock);

    caslock_acquire(&flusher_.lock);
    b->prev_live = NULL;
    b->next_live = flusher_.live_bufs;
    if (flusher_.live_bufs != NULL) {
        flusher_.live_bufs->prev_live = b;
    }
    flusher_.live_bufs = b;
    flusher_.nlive++;
    caslock_release(&flusher_.lock);
    return b;
}

static void
buffers_put_(struct writer_buffers *b)
{
    caslock_acquire(&b->lock);
    b->owner = NULL;
    caslock_release(&b->lock);

    caslock_acquire(&flusher_.lock);
    if (b->prev_live != NULL) {
        b->prev_live->next_live = b->next_live;
    } else {
        flusher_.live_bufs = b->next_live;
    }
    if (b->next_live != NULL) {
        b->next_live->prev_live = b->prev_live;
    }
    flusher_.nlive--;
    b->next_free       = flusher_.free_bufs;
    flusher_.free_bufs = b;
    caslock_release(&flusher_.lock);
}

static uint64_t *
buffer_alloc_(size_t size)
{
    void *p = coldtrace_mmap(NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) {
        log_warn("mmap trace buffer: %s", strerror(errno));
        return NULL;
    }
    return p;
}

// -----------------------------------------------------------------------------
// fragments
// -----------------------------------------------------------------------------

static void
flush_(struct writer_impl *impl)
{
    coldtrace_writer_close(impl->buffer, impl->offset, impl->md);

    if (coldtrace_writes_disabled()) {
        return;
    }

    const char *pattern = coldtrace_get_file_pattern();
    char file_name[strlen(pattern) + FORMAT_EXPANSION_SPACE];
    sprintf(file_name, pattern, impl->tid, impl->enumerator);

    int fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC, FILE_PERMISSIONS);
    if (fd == -1) {
        log_warn("open flush: %s", strerror(errno));
        impl->buffer = NULL;
        impl->failed = true;
        return;
    }

    if (!write_all_(fd, impl->buffer, impl->offset)) {
        log_warn("write flush: %s", strerror(errno));
        impl->buffer = NULL;
        impl->failed = true;
    }

    close(fd);
}

// Stamp the version header at the front of a fresh fragment and position
// offset just past it
static void
begin_fragment_(struct writer_impl *impl)
{
    struct version_header *header = (struct version_header *)impl->buffer;
    *header                       = current_version_header;
    impl->offset                  = sizeof(struct version_header);
}

// Hands the filled fragment to the flusher and switches to the other buffer.
// Falls back to writing the fragment synchronously.
static void
rotate_(struct writer_impl *impl)
{
    coldtrace_writer_close(impl->buffer, impl->offset, impl->md);

    struct writer_buffers *b = impl->bufs;
    caslock_acquire(&b->lock);
    struct flush_job *job = &b->job[b->cur];
    job->data             = impl->buffer;
    job->len              = impl->offset;
    job->pid              = getpid();
    snprintf(job->file_name, sizeof(job->file_name),
             coldtrace_get_file_pattern(), impl->tid, impl->enumerator);

    // two fragments in flight must not map to the same file
    unsigned next = b->cur;
    if (coldtrace_get_max() > 1 && flusher_enqueue_(job)) {
        unsigned other = b->cur ^ 1;
        if (b->buf[other] == NULL) {
            b->buf[other] = buffer_alloc_(impl->size);
        }
        if (b->buf[other] != NULL) {
            next = other;
        }
    } else {
        flush_job_write_(job);
    }

    // the fragment last written from the next buffer has to be on disk
    if (!flush_job_wait_(&b->job[next])) {
        impl->buffer = NULL;
        impl->failed = true;
    } else {
        b->cur           = next;
        impl->buffer     = b->buf[next];
        impl->enumerator = (impl->enumerator + 1) % coldtrace_get_max();
        begin_fragment_(impl);
    }
    caslock_release(&b->lock);
}

// Creates the file of the first fragment, skipping names that already exist
// (e.g., from an earlier writer of the same thread).
static bool
reserve_first_file_(struct writer_impl *impl)
{
    char file_name[FILE_NAME_SIZE];
    for (;;) {
        snprintf(file_name, sizeof(file_name), coldtrace_get_file_pattern(),
                 impl->tid, impl->enumerator);
        int fd = open(file_name, O_WRONLY | O_CREAT | O_EXCL, FILE_PERMISSIONS);
        if (fd != -1) {
            close(fd);
            return true;
        }
        if (errno != EEXIST) {
            log_warn("open get_trace: %s", strerror(errno));
            return false;
        }
        impl->enumerator++;
    }
}

static bool
ensure_buffer_(struct writer_impl *impl)
{
    if (!impl->initd) {
        log_warn("Writer not initialized (at %s:%d)", __FILE__, __LINE__);
        impl->buffer = NULL;
        return false;
    }
    if (impl->failed) {
        impl->buffer = NULL;
        return false;
    }
    if (impl->buffer) {
        return true;
    }

    impl->size       = coldtrace_get_trace_size();
    impl->offset     = 0;
    impl->enumerator = 0;

    if (coldtrace_writes_disabled()) {
        impl->buffer = mempool_alloc(impl->size);
        if (impl->buffer == NULL) {
            return false;
        }
        begin_fragment_(impl);
        return true;
    }

    if (!reserve_first_file_(impl)) {
        impl->failed = true;
        return false;
    }

    // Allocate the trace buffer once and reuse it for every fragment
    uint64_t *buffer = buffer_alloc_(impl->size);
    if (buffer == NULL) {
        impl->failed = true;
        return false;
    }
    impl->bufs = buffers_get_(impl);
    if (impl->bufs == NULL) {
        coldtrace_munmap(buffer, impl->size);
        impl->failed = true;
        return false;
    }
    caslock_acquire(&impl->bufs->lock);
    impl->bufs->buf[0] = impl->buffer = buffer;
    begin_fragment_(impl);
    caslock_release(&impl->bufs->lock);
    return true;
}

// Move to the next fragment after a flush with disabled writes
static void
advance_fragment_(struct writer_impl *impl)
{
    size_t trace_size = coldtrace_get_trace_size();

    if (impl->size != trace_size) {
        impl->size = trace_size;
        mempool_free(impl->buffer);
        impl->buffer = mempool_alloc(impl->size);
    }
    impl->offset = 0;
}

// Flush the current fragment and open a fresh one
static void
new_trace_(struct writer_impl *impl)
{
    if (!coldtrace_writes_disabled()) {
        rotate_(impl);
        return;
    }

    flush_(impl);
    if (impl->buffer == NULL) {
        return;
    }

    advance_fragment_(impl);
    if (impl->buffer == NULL) {
        return;
    }

    begin_fragment_(impl);
}

DICE_HIDE bool
coldtrace_writer_new_trace(struct coldtrace_writer *ct, size_t size)
{
    struct writer_impl *impl = (struct writer_impl *)ct;
    size_t sz                = coldtrace_get_trace_size();
    size_t trace_size        = impl->size;
    if (unlikely(sz < trace_size)) {
        trace_size = sz;
    }

    return (impl->offset + size) > trace_size;
}


static __attribute__((noinline)) void *
reserve_slow_(struct coldtrace_writer *ct, size_t size)
{
    struct writer_impl *impl = (struct writer_impl *)ct;
    if (!ensure_buffer_(impl)) {
        return NULL;
    }

    if (coldtrace_writer_new_trace(ct, size)) {
        new_trace_(impl);
        if (impl->buffer == NULL) {
            return NULL;
        }
    }

    char *ptr = (char *)(impl->buffer) + impl->offset;
    impl->offset += size;
    return ptr;
}

DICE_HIDE void *
coldtrace_writer_reserve(struct coldtrace_writer *ct, size_t size)
{
    struct writer_impl *impl = (struct writer_impl *)ct;
    // A non-NULL buffer implies an initialized writer that has not failed.
    if (likely(impl->buffer != NULL) &&
        likely(!coldtrace_writer_new_trace(ct, size))) {
        char *ptr = (char *)(impl->buffer) + impl->offset;
        impl->offset += size;
        return ptr;
    }
    return reserve_slow_(ct, size);
}

DICE_HIDE void
coldtrace_writer_init(struct coldtrace_writer *ct, metadata_t *md)
{
    struct writer_impl *impl = (struct writer_impl *)ct;
    if (md == NULL) {
        log_warn("No metadata provided (at %s:%d)", __FILE__, __LINE__);
        impl->initd  = false;
        impl->buffer = NULL;
        return;
    }
    impl->initd      = true;
    impl->failed     = false;
    impl->tid        = self_id(md);
    impl->buffer     = NULL;
    impl->offset     = 0;
    impl->enumerator = 0;
    impl->size       = 0;
    impl->md         = md;
    impl->bufs       = NULL;
}

DICE_HIDE void
coldtrace_writer_fini(struct coldtrace_writer *ct)
{
    struct writer_impl *impl = (struct writer_impl *)ct;
    if (!impl->initd) {
        return;
    }

    if (coldtrace_writes_disabled()) {
        if (impl->buffer != NULL) {
            flush_(impl);
            mempool_free(impl->buffer);
        }
        return;
    }

    // Nothing was ever traced on this thread
    struct writer_buffers *b = impl->bufs;
    if (b == NULL) {
        return;
    }

    // earlier fragments must reach their files before the buffers go
    bool ok0 = flush_job_wait_(&b->job[0]);
    bool ok1 = flush_job_wait_(&b->job[1]);

    // Flush the final partial fragment
    caslock_acquire(&b->lock);
    if (ok0 && ok1 && impl->buffer != NULL) {
        flush_(impl);
    }
    caslock_release(&b->lock);

    for (int i = 0; i < 2; i++) {
        if (b->buf[i] != NULL) {
            coldtrace_munmap(b->buf[i], impl->size);
        }
    }
    buffers_put_(b);
    impl->bufs   = NULL;
    impl->buffer = NULL;
}


__attribute__((weak)) void
coldtrace_writer_close(void *page, size_t size, metadata_t *md)
{
}

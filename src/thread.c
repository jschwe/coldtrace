/*
 * Copyright (C) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <coldtrace/config.h>
#include <coldtrace/libc.h>
#include <coldtrace/thread.h>
#include <coldtrace/writer.h>
#include <dice/log.h>
#include <dice/mempool.h>
#include <dice/module.h>
#include <dice/pubsub.h>
#include <dice/self.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vsync/atomic.h>

#define INITIAL_STACK_CAPACITY 16

struct coldtrace_thread {
    struct coldtrace_writer writer;
    bool initd;
    bool failed;
    uint32_t stack_bottom;
    uint64_t *stack;
    uint32_t stack_size;
    uint32_t stack_capacity;
    uint64_t created_thread_idx;
};

static struct coldtrace_thread tls_key_;

/* Two-level table caching the SELF_TLS lookup by dice thread id. Dice still
 * owns the thread objects; a slot is cleared when its thread is finalized. */
#define THREAD_CACHE_L2_BITS 10
#define THREAD_CACHE_L2_SIZE (1UL << THREAD_CACHE_L2_BITS)
#define THREAD_CACHE_L1_SIZE 1024

static vatomicptr_t thread_cache_[THREAD_CACHE_L1_SIZE];

static inline struct coldtrace_thread **
thread_cache_slot_(thread_id id, bool create)
{
    uint64_t hi = id >> THREAD_CACHE_L2_BITS;
    if (id == NO_THREAD || hi >= THREAD_CACHE_L1_SIZE) {
        return NULL;
    }
    struct coldtrace_thread **l2 = vatomicptr_read_acq(&thread_cache_[hi]);
    if (l2 == NULL && create) {
        size_t bytes = THREAD_CACHE_L2_SIZE * sizeof(*l2);
        l2           = mempool_alloc(bytes);
        if (l2 == NULL) {
            return NULL;
        }
        coldtrace_memset(l2, 0, bytes);
        void *cur = vatomicptr_cmpxchg_rel(&thread_cache_[hi], NULL, l2);
        if (cur != NULL) {
            mempool_free(l2);
            l2 = cur;
        }
    }
    return l2 ? &l2[id & (THREAD_CACHE_L2_SIZE - 1)] : NULL;
}

static __attribute__((noinline)) struct coldtrace_thread *
get_coldtrace_thread_slow_(metadata_t *md)
{
    struct coldtrace_thread *th = SELF_TLS(md, &tls_key_);
    if (!th->initd) {
        coldtrace_writer_init(&th->writer, md);
        th->initd = true;
    }
    struct coldtrace_thread **slot = thread_cache_slot_(self_id(md), true);
    if (slot) {
        *slot = th;
    }
    return th;
}

static inline struct coldtrace_thread *
get_coldtrace_thread(metadata_t *md)
{
    struct coldtrace_thread **slot = thread_cache_slot_(self_id(md), false);
    if (likely(slot != NULL && *slot != NULL)) {
        return *slot;
    }
    return get_coldtrace_thread_slow_(md);
}

static inline bool
with_stack_(coldtrace_entry_type type)
{
    switch (type & ~ZERO_FLAG) {
        case COLDTRACE_FREE:
        case COLDTRACE_ALLOC:
        case COLDTRACE_MMAP:
        case COLDTRACE_MUNMAP:
        case COLDTRACE_READ:
        case COLDTRACE_WRITE:
            return true;
        default:
            return false;
    }
}

static bool
grow_stack_(struct coldtrace_thread *th)
{
    uint32_t new_capacity;
    size_t bytes;

    if (th->stack_capacity == 0) {
        new_capacity = INITIAL_STACK_CAPACITY;
    } else {
        // we'd get a SEGV from stack overflow before this exceeds legal sizes
        new_capacity = th->stack_capacity * 2;
    }
    bytes = (size_t)new_capacity * sizeof(*th->stack);

    uint64_t *stack = coldtrace_realloc(th->stack, bytes);
    if (stack == NULL) {
        log_warn(
            "error: Could not increase thread stack size, disabling tracing");
        th->failed = true;
        return false;
    }

    th->stack          = stack;
    th->stack_capacity = new_capacity;
    return true;
}

DICE_HIDE void *
coldtrace_thread_append(struct metadata *md, coldtrace_entry_type type,
                        const void *ptr)
{
    struct coldtrace_thread *th = get_coldtrace_thread(md);
    if (th->failed) {
        return NULL;
    }

    uint64_t len = coldtrace_entry_fixed_size(type);
    if (len == 0) {
        log_warn("error: Unknown entry type %u, dropping entry", type);
        return NULL;
    }
    if (!with_stack_(type)) {
        if (coldtrace_writer_new_trace(&th->writer, len)) {
            th->stack_bottom = 0;
        }
        struct coldtrace_entry_header *entry =
            (struct coldtrace_entry_header *)coldtrace_writer_reserve(
                &th->writer, len);
        if (entry == NULL) {
            return NULL;
        }
        *entry = coldtrace_entry_init(type, ptr);
        return entry;
    }

    uint32_t stack_bot   = th->stack_bottom;
    uint32_t stack_top   = th->stack_size;
    uint64_t *stack_base = th->stack;
    size_t stack_size    = (size_t)(stack_top - stack_bot) * sizeof(uint64_t);
    if (coldtrace_writer_new_trace(&th->writer, len + stack_size)) {
        stack_bot  = 0;
        stack_size = (stack_top) * sizeof(uint64_t);
    }
    void *e = coldtrace_writer_reserve(&th->writer, len + stack_size);
    if (e == NULL) {
        log_warn("error: Could not reserve entry in writer, dropping entry");
        return NULL;
    }

    struct coldtrace_entry_header *entry = (struct coldtrace_entry_header *)e;
    *entry                               = coldtrace_entry_init(type, ptr);

    char *buf = (char *)e + len - sizeof(struct coldtrace_stack_diff);
    struct coldtrace_stack_diff *s = (struct coldtrace_stack_diff *)buf;
    s->depth                       = stack_top;
    s->popped                      = stack_bot;
    if (stack_size > 0) {
        coldtrace_memcpy(s->diff, stack_base + stack_bot, stack_size);
    }

    th->stack_bottom = stack_top;
    return e;
}

DICE_HIDE void
coldtrace_thread_init(struct metadata *md)
{
    struct coldtrace_thread *th = get_coldtrace_thread(md);
    coldtrace_writer_init(&th->writer, md);
}

DICE_HIDE void
coldtrace_thread_fini(struct metadata *md)
{
    struct coldtrace_thread *th = get_coldtrace_thread(md);
    coldtrace_writer_fini(&th->writer);
    coldtrace_free(th->stack);
    th->stack          = NULL;
    th->stack_size     = 0;
    th->stack_capacity = 0;
    th->stack_bottom   = 0;

    // dice releases the thread object after this, so drop it from the cache
    struct coldtrace_thread **slot = thread_cache_slot_(self_id(md), false);
    if (slot) {
        *slot = NULL;
    }
}

DICE_HIDE void
coldtrace_thread_set_create_idx(struct metadata *md, uint64_t idx)
{
    struct coldtrace_thread *th = get_coldtrace_thread(md);
    th->created_thread_idx      = idx;
}

DICE_HIDE uint64_t
coldtrace_thread_get_create_idx(struct metadata *md)
{
    struct coldtrace_thread *th = get_coldtrace_thread(md);
    return th->created_thread_idx;
}

DICE_HIDE void
coldtrace_thread_stack_push(struct metadata *md, void *caller)
{
    struct coldtrace_thread *th = get_coldtrace_thread(md);
    if (th->failed) {
        return;
    }
    if (th->stack_size == th->stack_capacity && !grow_stack_(th)) {
        return;
    }
    th->stack[th->stack_size++] = (uint64_t)(uintptr_t)caller;
}

DICE_HIDE void
coldtrace_thread_stack_pop(struct metadata *md)
{
    struct coldtrace_thread *th = get_coldtrace_thread(md);
    if (th->failed) {
        return;
    }
    if (th->stack_size > 0) {
        th->stack_size--;
        if (th->stack_bottom > th->stack_size) {
            th->stack_bottom = th->stack_size;
        }
    }
}

__attribute__((weak)) void
coldtrace_main_thread_fini()
{
}

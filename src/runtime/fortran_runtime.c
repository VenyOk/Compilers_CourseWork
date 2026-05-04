#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <pthread.h>

typedef void (*kmpc_micro)(void* global_tid, void* bound_tid, ...);

static pthread_mutex_t gomp_lock = PTHREAD_MUTEX_INITIALIZER;
static int32_t next_thread_id = 0;

typedef struct {
    kmpc_micro microtask;
    void* data;
    int32_t num_threads;
} thread_task_t;

static void* thread_worker(void* arg) {
    thread_task_t* task = (thread_task_t*)arg;
    int32_t tid = __atomic_fetch_add(&next_thread_id, 1, __ATOMIC_SEQ_CST);
    task->microtask(&tid, &tid, task->data);
    return NULL;
}

int32_t GOMP_global_thread_num(void* ident) {
    static __thread int32_t tls_tid = -1;
    if (tls_tid == -1) {
        tls_tid = __atomic_fetch_add(&next_thread_id, 1, __ATOMIC_SEQ_CST);
    }
    return tls_tid;
}

int32_t __kmpc_global_thread_num(void* ident) {
    return GOMP_global_thread_num(ident);
}

void GOMP_parallel_start(void* fn, void* data, unsigned num_threads) {
    if (num_threads <= 1 || num_threads == 0) {
        int32_t tid = 0;
        ((kmpc_micro)fn)(&tid, &tid, data);
        return;
    }
    
    pthread_t* threads = malloc(sizeof(pthread_t) * (num_threads - 1));
    thread_task_t* tasks = malloc(sizeof(thread_task_t) * (num_threads - 1));
    
    for (unsigned i = 0; i < num_threads - 1; i++) {
        tasks[i].microtask = (kmpc_micro)fn;
        tasks[i].data = data;
        tasks[i].num_threads = num_threads;
        pthread_create(&threads[i], NULL, thread_worker, &tasks[i]);
    }
    
    int32_t master_tid = 0;
    ((kmpc_micro)fn)(&master_tid, &master_tid, data);
    
    for (unsigned i = 0; i < num_threads - 1; i++) {
        pthread_join(threads[i], NULL);
    }
    
    free(threads);
    free(tasks);
}

void __kmpc_fork_call(void* ident, int32_t argc, kmpc_micro microtask, ...) {
    void* args[argc];
    va_list ap;
    va_start(ap, microtask);
    for (int32_t i = 0; i < argc; i++) {
        args[i] = va_arg(ap, void*);
    }
    va_end(ap);
    
    int32_t tid = 0;
    microtask(&tid, &tid, args);
}

void GOMP_parallel_end(void) {
    __atomic_store_n(&next_thread_id, 0, __ATOMIC_SEQ_CST);
}

void __kmpc_barrier(void* ident, int32_t tid) {
}

void GOMP_barrier(void) {
}

int32_t __kmpc_master(void* ident, int32_t tid) {
    return 1;
}

void __kmpc_end_master(void* ident, int32_t tid) {
}

int32_t __kmpc_single(void* ident, int32_t tid) {
    return 1;
}

void __kmpc_end_single(void* ident, int32_t tid) {
}

void __kmpc_critical(void* ident, int32_t tid, void* lock) {
    pthread_mutex_lock(&gomp_lock);
}

void __kmpc_end_critical(void* ident, int32_t tid, void* lock) {
    pthread_mutex_unlock(&gomp_lock);
}

void GOMP_critical_start(void) {
    pthread_mutex_lock(&gomp_lock);
}

void GOMP_critical_end(void) {
    pthread_mutex_unlock(&gomp_lock);
}

void __kmpc_init_lock(void* lock) {
    pthread_mutex_init((pthread_mutex_t*)lock, NULL);
}

void __kmpc_destroy_lock(void* lock) {
    pthread_mutex_destroy((pthread_mutex_t*)lock);
}

void __kmpc_set_lock(void* lock) {
    pthread_mutex_lock((pthread_mutex_t*)lock);
}

void __kmpc_unset_lock(void* lock) {
    pthread_mutex_unlock((pthread_mutex_t*)lock);
}

int32_t __kmpc_test_lock(void* lock) {
    return pthread_mutex_trylock((pthread_mutex_t*)lock) == 0 ? 1 : 0;
}

void GOMP_loop_static_start(void* data, long start, long end, long stride, long* i_start, long* i_end) {
    *i_start = start;
    *i_end = end;
}

int32_t GOMP_loop_static_next(void* data, long* i_start, long* i_end) {
    return 0;
}

void GOMP_loop_end(void) {
}

void GOMP_loop_end_nowait(void) {
}

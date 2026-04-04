#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <process.h>
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#define FORTRAN_RUNTIME_EXPORT __declspec(dllexport)
#else
#define FORTRAN_RUNTIME_EXPORT
#define FORTRAN_TLS __thread
#endif

typedef void (*fortran_parallel_microtask)(void *);

typedef struct {
    fortran_parallel_microtask microtask;
    void *env;
    int32_t thread_num;
    int32_t num_threads;
} fortran_parallel_region_task;

#if defined(_WIN32)
static DWORD fortran_tls_thread_num_index = TLS_OUT_OF_INDEXES;
static DWORD fortran_tls_num_threads_index = TLS_OUT_OF_INDEXES;
static INIT_ONCE fortran_tls_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK fortran_tls_init_callback(PINIT_ONCE init_once, PVOID parameter, PVOID *context) {
    (void)init_once;
    (void)parameter;
    (void)context;
    fortran_tls_thread_num_index = TlsAlloc();
    fortran_tls_num_threads_index = TlsAlloc();
    return fortran_tls_thread_num_index != TLS_OUT_OF_INDEXES && fortran_tls_num_threads_index != TLS_OUT_OF_INDEXES;
}

static int fortran_tls_ready(void) {
    return InitOnceExecuteOnce(&fortran_tls_once, fortran_tls_init_callback, NULL, NULL) ? 1 : 0;
}

static int32_t fortran_tls_get_thread_num(void) {
    if (!fortran_tls_ready()) {
        return 0;
    }
    return (int32_t)(intptr_t)TlsGetValue(fortran_tls_thread_num_index);
}

static int32_t fortran_tls_get_num_threads(void) {
    if (!fortran_tls_ready()) {
        return 1;
    }
    return (int32_t)(intptr_t)TlsGetValue(fortran_tls_num_threads_index);
}

static void fortran_tls_set(int32_t thread_num, int32_t num_threads) {
    if (!fortran_tls_ready()) {
        return;
    }
    TlsSetValue(fortran_tls_thread_num_index, (LPVOID)(intptr_t)thread_num);
    TlsSetValue(fortran_tls_num_threads_index, (LPVOID)(intptr_t)num_threads);
}
#else
static FORTRAN_TLS int32_t fortran_tls_thread_num = 0;
static FORTRAN_TLS int32_t fortran_tls_num_threads = 1;

static int32_t fortran_tls_get_thread_num(void) {
    return fortran_tls_thread_num;
}

static int32_t fortran_tls_get_num_threads(void) {
    return fortran_tls_num_threads;
}

static void fortran_tls_set(int32_t thread_num, int32_t num_threads) {
    fortran_tls_thread_num = thread_num;
    fortran_tls_num_threads = num_threads;
}
#endif

static int32_t fortran_env_threads(void) {
#if defined(_WIN32)
    char buffer[32];
    DWORD length = GetEnvironmentVariableA("FORTRAN_PARALLEL_THREADS", buffer, (DWORD)sizeof(buffer));
    if (length == 0 || length >= sizeof(buffer)) {
        return 0;
    }
    buffer[length] = '\0';
    return (int32_t)strtol(buffer, NULL, 10);
#else
    const char *value = getenv("FORTRAN_PARALLEL_THREADS");
    if (value == NULL || *value == '\0') {
        return 0;
    }
    return (int32_t)strtol(value, NULL, 10);
#endif
}

static int32_t fortran_cpu_threads(void) {
#if defined(_WIN32)
    DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (count == 0) {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        count = info.dwNumberOfProcessors;
    }
    return (int32_t)(count > 0 ? count : 1);
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count <= 0) {
        return 1;
    }
    return (int32_t)count;
#endif
}

static int32_t fortran_choose_threads(int32_t requested_threads) {
    int32_t env_threads = fortran_env_threads();
    int32_t cpu_threads = fortran_cpu_threads();
    int32_t chosen = 1;
    if (requested_threads > 0 && env_threads > 0) {
        chosen = requested_threads < env_threads ? requested_threads : env_threads;
    } else if (requested_threads > 0) {
        chosen = requested_threads;
    } else if (env_threads > 0) {
        chosen = env_threads;
    } else {
        chosen = cpu_threads;
        if (chosen > 8) {
            chosen = 8;
        }
    }
    if (chosen < 1) {
        chosen = 1;
    }
    if (cpu_threads > 0 && chosen > cpu_threads) {
        chosen = cpu_threads;
    }
    return chosen;
}

static void fortran_run_microtask(fortran_parallel_microtask microtask, void *env, int32_t thread_num, int32_t num_threads) {
    int32_t old_thread_num = fortran_tls_get_thread_num();
    int32_t old_num_threads = fortran_tls_get_num_threads();
    fortran_tls_set(thread_num, num_threads);
    microtask(env);
    fortran_tls_set(old_thread_num, old_num_threads);
}

#if defined(_WIN32)
static unsigned __stdcall fortran_parallel_region_thread(void *param) {
    fortran_parallel_region_task *task = (fortran_parallel_region_task *)param;
    fortran_run_microtask(task->microtask, task->env, task->thread_num, task->num_threads);
    return 0;
}
#else
static void *fortran_parallel_region_thread(void *param) {
    fortran_parallel_region_task *task = (fortran_parallel_region_task *)param;
    fortran_run_microtask(task->microtask, task->env, task->thread_num, task->num_threads);
    return NULL;
}
#endif

FORTRAN_RUNTIME_EXPORT int32_t fortran_thread_num(void) {
    return fortran_tls_get_thread_num();
}

FORTRAN_RUNTIME_EXPORT int32_t fortran_num_threads(void) {
    return fortran_tls_get_num_threads();
}

FORTRAN_RUNTIME_EXPORT void fortran_parallel_region_launch(int32_t requested_threads, void *microtask_ptr, void *env_ptr) {
    int32_t num_threads;
    int32_t worker_count;
    int32_t index;
    fortran_parallel_microtask microtask;
    fortran_parallel_region_task *tasks;
    unsigned char *started;
#if defined(_WIN32)
    HANDLE *handles;
#else
    pthread_t *threads;
#endif

    if (microtask_ptr == NULL) {
        return;
    }

    microtask = (fortran_parallel_microtask)microtask_ptr;
    num_threads = fortran_choose_threads(requested_threads);
    if (num_threads <= 1) {
        fortran_run_microtask(microtask, env_ptr, 0, 1);
        return;
    }

    worker_count = num_threads - 1;
    tasks = (fortran_parallel_region_task *)calloc((size_t)worker_count, sizeof(fortran_parallel_region_task));
    started = (unsigned char *)calloc((size_t)worker_count, sizeof(unsigned char));
    if (tasks == NULL || started == NULL) {
        free(tasks);
        free(started);
        fortran_run_microtask(microtask, env_ptr, 0, 1);
        return;
    }

#if defined(_WIN32)
    handles = (HANDLE *)calloc((size_t)worker_count, sizeof(HANDLE));
    if (handles == NULL) {
        free(tasks);
        free(started);
        fortran_run_microtask(microtask, env_ptr, 0, 1);
        return;
    }
#else
    threads = (pthread_t *)calloc((size_t)worker_count, sizeof(pthread_t));
    if (threads == NULL) {
        free(tasks);
        free(started);
        fortran_run_microtask(microtask, env_ptr, 0, 1);
        return;
    }
#endif

    for (index = 0; index < worker_count; ++index) {
        tasks[index].microtask = microtask;
        tasks[index].env = env_ptr;
        tasks[index].thread_num = index + 1;
        tasks[index].num_threads = num_threads;
#if defined(_WIN32)
        handles[index] = (HANDLE)_beginthreadex(NULL, 0, fortran_parallel_region_thread, &tasks[index], 0, NULL);
        if (handles[index] != NULL) {
            started[index] = 1;
        }
#else
        if (pthread_create(&threads[index], NULL, fortran_parallel_region_thread, &tasks[index]) == 0) {
            started[index] = 1;
        }
#endif
    }

    fortran_run_microtask(microtask, env_ptr, 0, num_threads);

    for (index = 0; index < worker_count; ++index) {
#if defined(_WIN32)
        if (started[index]) {
            WaitForSingleObject(handles[index], INFINITE);
            CloseHandle(handles[index]);
        } else {
            fortran_run_microtask(microtask, env_ptr, tasks[index].thread_num, tasks[index].num_threads);
        }
#else
        if (started[index]) {
            pthread_join(threads[index], NULL);
        } else {
            fortran_run_microtask(microtask, env_ptr, tasks[index].thread_num, tasks[index].num_threads);
        }
#endif
    }

#if defined(_WIN32)
    free(handles);
#else
    free(threads);
#endif
    free(tasks);
    free(started);
}

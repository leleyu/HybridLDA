//#include "csr.h"
#include "lda_jni_Utils.h"
#include "csr.h"
#include <omp.h>
#include <random>
#include <math.h>
#include <string.h>
#include <sys/time.h>

class xorshift
{
	uint64_t s[16];
	uint32_t p;
	uint64_t x; /* The state must be seeded with a nonzero value. */
    uint32_t max_int;

	uint64_t xorshift1024star(void) {
		uint64_t s0 = s[ p ];
		uint64_t s1 = s[ p = (p+1) & 15 ];
		s1 ^= s1 << 31; // a
		s1 ^= s1 >> 11; // b
		s0 ^= s0 >> 30; // c
		return ( s[p] = s0 ^ s1 ) * UINT64_C(1181783497276652981);
	}
	uint64_t xorshift128plus(void) {
		uint64_t x = s[0];
		uint64_t const y = s[1];
		s[0] = y;
		x ^= x << 23; // a
		x ^= x >> 17; // b
		x ^= y ^ (y >> 26); // c
		s[1] = x;
		return x + y;
	}
	uint64_t xorshift64star(void) {
		x ^= x >> 12; // a
		x ^= x << 25; // b
		x ^= x >> 27; // c
		return x * UINT64_C(2685821657736338717);
	}
	public:

	using result_type=uint64_t;

	xorshift() : p(0), x((uint64_t)std::rand() * RAND_MAX + std::rand()){
		for (uint32_t i = 0; i < 16; i++)
		{
			s[i] = xorshift64star();
		}

    max_int = std::numeric_limits<uint32_t>::max();
	}
	uint64_t operator()(){
		return xorshift128plus();
	}
	uint32_t Rand32(){
		return (uint32_t) xorshift128plus();
	}
    
    uint32_t rand_int(uint32_t max) {
        return Rand32() % max;
    }
    
    float rand_double() {
        return Rand32() * 1.0F / max_int;
    }
    
    
	void MakeBuffer(void *p, size_t len)
	{
		uint32_t N = (int) len / sizeof(uint32_t);
		uint32_t *arr = (uint32_t *)p;
		for (uint32_t i = 0; i < N; i++)
			arr[i] = (uint32_t)(*this)();
		uint32_t M = len % sizeof(uint32_t);
		if (M > 0)
		{
			uint32_t k = (uint32_t)(*this)();
			memcpy(arr + N, &k, M);
		}
	}
	uint64_t max() {return std::numeric_limits<uint64_t>::max();}
	uint64_t min() {return std::numeric_limits<uint64_t>::min();}
};

inline static long get_current_ms() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    
    long timestamp = (long) tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return timestamp;
}

inline static std::mt19937 & urng() {
    static std::mt19937 u{};
    return u;
}

inline static double unif01() {
    static std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(urng());
}

inline static uint32_t rand_int(int from, int to) {
    static std::uniform_int_distribution<> d{};
    using param_t = std::uniform_int_distribution<>::param_type;
    return d(urng(), param_t{from, to});
}

void init_warp(int D, int V, int K,
        int * ws, int * ds, int * widx, int * nk,
        short * topics, short * mhs,
        xorshift&  gen) {
    
    short tt;
    for (int w = 0; w < V; w ++) {
        for (int wi = ws[w]; wi < ws[w + 1]; wi ++) {
            tt = (short)gen.rand_int(K);
            topics[wi] = tt;
            nk[tt] ++;
            for (int m = 0; m < MH_STEPS; m ++)
                mhs[wi * MH_STEPS + m] = tt;
        }
    }
}

bool TEST = false;

double visit_by_row(int D, int V, int K,
        int * ds, int * widx, int * nk,
        int * tk, int * nidx, int * used,
        short * topics, short * mhs,
        float alpha, float beta, float vbeta,
        xorshift* gen,
        int max_len, short * local_topics, short * local_mhs) {
    short tt, ttt;
    short k;
    int N, idx;
    float p;

    double lgamma_alpha = lgamma(alpha);
    double ll = 0;
    
    // traverse each doc in sequential order
    for (int d = 0; d < D; d ++) {
        N = ds[d + 1] - ds[d];
        if (N == 0)
            continue;

        // compute dk for doc on the fly
        memset(tk, 0, K * sizeof(int));
        //memset(used, 0, K * sizeof(int));

        int di = ds[d];
        idx = 0;
        for (int i = 0; i < N; i ++) {
            k = topics[widx[di + i]];
            local_topics[i] = k;
            tk[k] ++;
            if (used[k] == 0) {
                used[k] = 1;
                nidx[idx ++] = k;
            }
                
            for (int m = 0; m < MH_STEPS; m ++)
                local_mhs[i * MH_STEPS + m] = mhs[widx[di + i] * MH_STEPS + m];
        }
        
        // compute llhw
        for (int i = 0; i < idx; i ++) {
            ll += lgamma(alpha + tk[nidx[i]]) - lgamma_alpha;
            used[nidx[i]] = 0;
        }

        ll -= lgamma(alpha * K + N) - lgamma(alpha * K);

        // traverse doc first time, accept word proposal
        for (int i = 0; i < N; i ++) {
            tt = local_topics[i];
            nk[tt] --;
            tk[tt] --;

            float b = tk[tt] + alpha;
            float d = nk[tt] + vbeta;

            for (unsigned m = 0; m < MH_STEPS; m ++) {
                ttt = local_mhs[i * MH_STEPS + m];
                float a = tk[ttt] + alpha;
                float c = nk[ttt] + vbeta;

                float ad = a * d;
                float bc = b * c;
                bool accept = gen->Rand32() * bc < ad * std::numeric_limits<uint32_t>::max();

                if (accept) {
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }

            nk[tt] ++;
            tk[tt] ++;
            local_topics[i] = tt;
        }

        // traverse doc second time, generate doc proposal
        double new_topic = alpha * K / (alpha * K + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * new_topic;
        for (int i = 0; i < N; i ++) {
            topics[widx[di + i]] = local_topics[i];
            for (int m = 0; m < MH_STEPS; m ++) {
                uint32_t si = widx[di + i] * MH_STEPS + m;
                if (gen->Rand32() < new_topic_th) {
                    mhs[si] = gen->Rand32() % K;
                } else {
                    mhs[si] = local_topics[gen->Rand32() % N];
                }
            }
        }
    }
    return ll;
}

double visit_by_col(int D, int V, int K,
        int * ws, int * nk,
        int * tk, int * nidx, int * used,
        short * topics, short * mhs,
        float alpha, float beta, float vbeta,
        xorshift * gen) {

    short tt, ttt;
    short k;
    int N, idx;
    float p;

    double lgamma_beta = lgamma(beta);
    double ll = 0.0;

    for (int w = 0; w < V; w ++) {
        N = ws[w + 1] - ws[w];

        if (N == 0)
            continue;

        // compute wk for word on the fly
        memset(tk, 0, K * sizeof(int));
        //memset(used, 0, K * sizeof(int));
        
        idx = 0;
        for (int wi = ws[w]; wi < ws[w + 1]; wi ++) {
            k = topics[wi];
            tk[k] ++;
            if (used[k] == 0) {
                nidx[idx ++] = k;
                used[k] = 1;
            }
        }
            
        // compute llhw
        for (int i = 0; i < idx; i ++) {
            ll += lgamma(beta + tk[nidx[i]]) - lgamma_beta;
            used[nidx[i]] = 0; 
        }

        // traverse w first time, accept doc proposal
        for (int wi = ws[w]; wi < ws[w + 1]; wi ++) {
            tt = topics[wi];
            tk[tt] --;
            nk[tt] --;
            
            float b = tk[tt]  + beta;
            float d = nk[tt]  + vbeta;
            
            for (unsigned m = 0; m < MH_STEPS; m ++) {
                ttt = mhs[wi * MH_STEPS + m];
                float a = tk[ttt] + beta;
                float c = nk[ttt] + vbeta;
                bool accept = gen->Rand32() * b* c < a * d * std::numeric_limits<uint32_t>::max();
                if (accept) {
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }
            
            tk[tt] ++;
            nk[tt] ++;
            topics[wi] = tt;
        }

        // traverse w second time, compute word proposal
        p = (K * beta) / (K * beta + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * p;
        for (int wi = ws[w]; wi < ws[w + 1]; wi ++) {
            for (int m = 0; m < MH_STEPS; m ++) {
                if (gen->Rand32() < new_topic_th) {
                    mhs[wi * MH_STEPS + m] = gen->rand_int(K);
                } else {
                    mhs[wi * MH_STEPS + m] = topics[ws[w] + gen->Rand32() % N];
                }
            }
        }
    }

    for (int k = 0; k < K; k ++) {
        if (nk[k] > 0)
          ll += -lgamma(nk[k] + vbeta) + lgamma(vbeta);
    }

    return ll;

}

double train_for_one_iter(int iter, int D, int V, int K, int N,
        int * ws, int * ds, int * widx, int * nk,
        int * tk, int * nidx, int * used,
        short * topics, short * mhs,
        float alpha, float beta, float vbeta,
        int max_len, short * local_topics, short * local_mhs) {

    xorshift gen;

    double ll = 0;
    long start, end, stt;
    long row_tt, col_tt, iter_tt;

    start = get_current_ms();
    stt = start;
    ll += visit_by_col(D, V, K, ws, nk, tk, nidx, used,
            topics, mhs, alpha, beta, vbeta, &gen);
    end = get_current_ms();
    col_tt = end - start;

    start = get_current_ms();
    ll += visit_by_row(D, V, K, ds, widx, nk, tk, nidx, used,
            topics, mhs, alpha, beta, vbeta, &gen,
            max_len, local_topics, local_mhs);
    end = get_current_ms();
    row_tt = end - start;
    iter_tt = end - stt;

    printf("iter=%d iter_tt=%ld col_tt=%ld row_tt=%ld ll=%f %f M tokens/s\n", 
            iter, iter_tt, col_tt, row_tt, ll, ((double) N/1e6) / (iter_tt / 1e3));
    return ll;
}

const static int THREAD_NUM = 16;

struct thread_params {
    xorshift gen;
    int * tk;
    int * nidx;
    int * used;
    int * nk;
    int * nk_new;
    short * local_topics;
    short * local_mhs;

    double ll;
    double one_over_accept_sum;
    long sample_times;
    long accept_times;
    int K;

    thread_params() {}
    
    void init(int K, int max_len, int mhstep = MH_STEPS) {
        tk = new int[K];
        nidx = new int[K];
        used = new int[K];
        nk = new int[K];
        nk_new = new int[K];
        local_topics = new short[max_len];
        local_mhs    = new short[max_len * mhstep];
        ll = 0;
        this->K = K;
        memset(used, 0, K * sizeof(int));
        memset(tk, 0, K * sizeof(int));
        one_over_accept_sum = 0.0;
        sample_times = 0;
        accept_times = 0;
    }

    ~thread_params() {
        delete [] tk;
        delete [] nidx;
        delete [] used;
        delete [] nk;
        delete [] nk_new;
        delete [] local_topics;
        delete [] local_mhs;
    }

};

void parallel_reduce_nk(int * global_nk, thread_params * params, int K) {
#pragma omp parallel for num_threads(THREAD_NUM)
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < THREAD_NUM; i ++)
            global_nk[k] += params[i].nk_new[k];
    }
}


void init_warp_parallel(int D, int V, int K,
        int * ws, int * ds, int * widx, int * global_nk,
        short * topics, short * mhs,
        thread_params * params) {
    
    for (int k = 0; k < K; k ++)
        global_nk[k] = 0;

    for (int i = 0; i < THREAD_NUM; i ++)
        for (int k = 0; k < K; k ++)
            params[i].nk_new[k] = 0;

#pragma omp parallel for num_threads(THREAD_NUM) 
    for (int w = 0; w < V; w ++) {
        int tn = omp_get_thread_num();
        int* nk_new = params[tn].nk_new;
        for (int wi = ws[w]; wi < ws[w + 1]; wi ++) {
            short tt = (short) params[tn].gen.rand_int(K);
            topics[wi] = tt;
            nk_new[tt] ++;
            for (int m = 0; m < MH_STEPS; m ++)
                mhs[wi * MH_STEPS + m] = tt;
        }
    }

    parallel_reduce_nk(global_nk, params, K);

    for (int i = 0; i < 10; i ++)
        printf("%d ", global_nk[i]);
    printf("\n");
}

void parallel_copy_nk(int * global_nk, thread_params * params, int K) {
#pragma omp parallel for num_threads(THREAD_NUM) 
    for (int k = 0; k < K; k ++) {
        for (int i = 0; i < THREAD_NUM; i ++)
            params[i].nk[k] = global_nk[k];
    }
    
    for (int i = 0; i < THREAD_NUM; i ++)
        memset(params[i].nk_new, 0, sizeof(int) * K);
}

double visit_by_row_parallel(int D, int V, int K,
        int * ds, int * widx, int * global_nk,
        bool * is_short,
        short * topics, short * mhs, 
        float alpha, float beta, float vbeta,
        thread_params * params) {

    double lgamma_alpha = lgamma(alpha);
    double global_ll = 0;

    parallel_copy_nk(global_nk, params, K);
    
    // traverse each doc
#pragma omp parallel for num_threads(THREAD_NUM)
    for (int d = 0; d < D; d ++) {
        int tn = omp_get_thread_num();
        int N = ds[d + 1] - ds[d];
        if (N == 0)
            continue;

        if (is_short[d])
            continue;

        short tt, ttt, k;
        int idx;
        float p;


        int* tk = params[tn].tk;
        int* nidx = params[tn].nidx;
        int* used = params[tn].used;
        int* nk = params[tn].nk;
        int* nk_new = params[tn].nk_new;
        short* local_topics = params[tn].local_topics;
        short* local_mhs = params[tn].local_mhs;
        double one_over_accept_local = 0.0;
        long sample_times_local = 0;
        long accept_times_local = 0;

        // compute dk for doc on the fly
        memset(tk, 0, K * sizeof(int));
        //memset(used, 0, K * sizeof(int));
        
        idx = 0;
        int di = ds[d];
        for (int i = 0; i < N; i ++) {
            k = topics[widx[di + i]];
            local_topics[i] = k;
            tk[k] ++;
            if (used[k] == 0) {
                used[k] = 1;
                nidx[idx ++] = k;
            }

            for (int m = 0; m < MH_STEPS; m ++)
                local_mhs[i * MH_STEPS + m] = mhs[widx[di + i] * MH_STEPS + m];
        }

        //if (idx < 64)
        //    is_short[d] = true;

        // compute llhw
        for (int i = 0; i < idx; i ++) {
            params[tn].ll += lgamma(alpha + tk[nidx[i]]) - lgamma_alpha;
            used[nidx[i]] = 0;
        }


        params[tn].ll -= lgamma(alpha * K + N) - lgamma(alpha * K);

        // traverse doc first time, accept word proposal
        for (int i = 0; i < N; i ++) {
            tt = local_topics[i];
            nk[tt] --;
            tk[tt] --;
            nk_new[tt] --;

            float b = tk[tt] + alpha;
            float d = nk[tt] + vbeta;

            for (unsigned m = 0; m < MH_STEPS; m ++) {
                ttt = local_mhs[i * MH_STEPS + m];
                float a = tk[ttt] + alpha;
                float c = nk[ttt] + vbeta;

                float ad = a * d;
                float bc = b * c;
                bool accept = params[tn].gen.Rand32() * bc < ad * std::numeric_limits<uint32_t>::max();
                float ac = ad / bc;
                if (ac > 1) ac = 1;
                one_over_accept_local += ac;
                sample_times_local += 1;

                if (accept) {
                    accept_times_local += 1;
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }

            nk[tt] ++;
            tk[tt] ++;
            local_topics[i] = tt;
            nk_new[tt] ++;
        }
        // traverse doc second time, generate doc proposal
        double new_topic = alpha * K / (alpha * K + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * new_topic;
        for (int i = 0; i < N; i ++) {
            topics[widx[di+i]] = local_topics[i];
            for (int m = 0; m < MH_STEPS; m ++) {
                uint32_t si = widx[di + i] * MH_STEPS + m;
                if (params[tn].gen.Rand32() < new_topic_th) {
                    mhs[si] = params[tn].gen.Rand32() % K;
                } else {
                    mhs[si] = local_topics[params[tn].gen.Rand32() % N];
                }
            }
        }

        params[tn].one_over_accept_sum += one_over_accept_local;
        params[tn].sample_times += sample_times_local;
        params[tn].accept_times += accept_times_local;
    }

    parallel_reduce_nk(global_nk, params, K);

    for (int i = 0; i < THREAD_NUM; i ++)
        global_ll += params[i].ll;

    double one_over_accept_all = 0;
    long sample_times_all = 0;
    long accept_times_all = 0;
    for (int i = 0; i < THREAD_NUM; i ++) {
        one_over_accept_all += params[i].one_over_accept_sum;
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }
    printf("one_over_accept_all=%f sample_times_all=%ld accept_times_all=%ld one_over_accept=%f accept_per=%f\n",
            one_over_accept_all, sample_times_all, accept_times_all,
            one_over_accept_all/sample_times_all,
            accept_times_all * 1.0 / sample_times_all);
    return global_ll;
}

double visit_by_col_parallel(int D, int V, int K,
        int * ws, int * global_nk,
        short * topics, short * mhs,
        float alpha, float beta, float vbeta,
        thread_params * params) {

    double lgamma_beta = lgamma(beta);
    double global_ll = 0.0;

    parallel_copy_nk(global_nk, params, K);

#pragma omp parallel for num_threads(THREAD_NUM)
    for (int w = 0; w < V; w ++) {
        int tn = omp_get_thread_num();
        int N = ws[w + 1] - ws[w];
        short tt, ttt, k;
        int idx;
        float p;

        if (N == 0)
            continue;

        int wi = ws[w];
        int * tk = params[tn].tk;
        int * nidx = params[tn].nidx;
        int * used = params[tn].used;
        int * nk = params[tn].nk;
        int * nk_new = params[tn].nk_new;
        short * local_topics = topics + wi;
        short * local_mhs    = mhs + wi * MH_STEPS;
        double one_over_accept_local = 0.0;
        long sample_times_local = 0;
        long accept_times_local = 0;

        // compute wk for word on the fly
        memset(tk, 0, K * sizeof(int));
        
        idx = 0;
        for (int i = 0; i < N; i ++) {
            k = local_topics[i];
            tk[k] ++;
            if (used[k] == 0) {
                nidx[idx ++] = k;
                used[k] = 1;
            }
        }
            
        // compute llhw
        for (int i = 0; i < idx; i ++) {
            params[tn].ll += lgamma(beta + tk[nidx[i]]) - lgamma_beta;
            used[nidx[i]] = 0; 
        }

        // traverse w first time, accept doc proposal
        for (int i = 0; i < N; i ++) {
            tt = local_topics[i];
            if (local_mhs[i * MH_STEPS] < 0)
                continue;

            tk[tt] --;
            nk[tt] --;
            nk_new[tt] --;
            
            float b = tk[tt]  + beta;
            float d = nk[tt]  + vbeta;
            
            for (unsigned m = 0; m < MH_STEPS; m ++) {
                ttt = local_mhs[i * MH_STEPS + m];
                float a = tk[ttt] + beta;
                float c = nk[ttt] + vbeta;
                bool accept = params[tn].gen.Rand32() * b* c < a * d * std::numeric_limits<uint32_t>::max();
                float ac = a * d / (b*c);
                if (ac > 1) ac = 1;
                one_over_accept_local +=  ac;
                sample_times_local += 1;

                if (accept) {
                    accept_times_local += 1;
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }
            
            tk[tt] ++;
            nk[tt] ++;
            nk_new[tt] ++;
            local_topics[i] = tt;
        }

        // traverse w second time, compute word proposal
        p = (K * beta) / (K * beta + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * p;
        for (int i = 0; i < N; i ++) {
            if (local_mhs[i * MH_STEPS] < 0)
                continue;
            for (int m = 0; m < MH_STEPS; m ++) {
                uint32_t si = i * MH_STEPS + m;
                if (params[tn].gen.Rand32() < new_topic_th) {
                    local_mhs[si] = params[tn].gen.rand_int(K);
                } else {
                    local_mhs[si] = local_topics[params[tn].gen.Rand32() % N];
                }
            }
        }

        params[tn].one_over_accept_sum += one_over_accept_local;
        params[tn].sample_times += sample_times_local;
        params[tn].accept_times += accept_times_local;

    }

    parallel_reduce_nk(global_nk, params, K);
    
    for (int k = 0; k < K; k ++) {
        if (global_nk[k] > 0)
          global_ll += - lgamma(global_nk[k] + vbeta) + lgamma(vbeta);
    }
    
    for (int i = 0; i < THREAD_NUM; i ++)
        global_ll += params[i].ll;

    double one_over_accept_all = 0;
    long sample_times_all = 0;
    long accept_times_all = 0;
    for (int i = 0; i < THREAD_NUM; i ++) {
        one_over_accept_all += params[i].one_over_accept_sum;
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }
    printf("one_over_accept_all=%f sample_times_all=%ld accept_times_all=%ld one_over_accept=%f accept_per=%f\n",
            one_over_accept_all, sample_times_all, accept_times_all,
            one_over_accept_all/sample_times_all,
            accept_times_all * 1.0 / sample_times_all);

    return global_ll;
}

double visit_by_row_parallel_seperate(int ld, int V, int K,
        int * ds, int * widx, int * global_nk,
        bool * is_short,
        short * topics, short * mhs, 
        float alpha, float beta, float vbeta,
        thread_params * params, int mhstep) {

    double lgamma_alpha = lgamma(alpha);
    double global_ll = 0;

    parallel_copy_nk(global_nk, params, K);
    
    // traverse each doc
#pragma omp parallel for num_threads(THREAD_NUM)
    for (int d = 0; d < ld; d ++) {
        int tn = omp_get_thread_num();
        int N = ds[d + 1] - ds[d];
        if (N == 0)
            continue;

        short tt, ttt, k;
        int idx;
        float p;


        int* tk = params[tn].tk;
        int* nidx = params[tn].nidx;
        int* used = params[tn].used;
        int* nk = params[tn].nk;
        int* nk_new = params[tn].nk_new;
        short* local_topics = params[tn].local_topics;
        short* local_mhs = params[tn].local_mhs;
        double one_over_accept_local = 0.0;
        long sample_times_local = 0;
        long accept_times_local = 0;

        // compute dk for doc on the fly
        memset(tk, 0, K * sizeof(int));
        //memset(used, 0, K * sizeof(int));
        
        idx = 0;
        int di = ds[d];
        for (int i = 0; i < N; i ++) {
            k = topics[widx[di + i]];
            local_topics[i] = k;
            tk[k] ++;
            if (used[k] == 0) {
                used[k] = 1;
                nidx[idx ++] = k;
            }

            for (int m = 0; m < mhstep; m ++)
                local_mhs[i * mhstep + m] = mhs[widx[di + i] * mhstep + m];
        }

        //if (idx < 64)
        //    is_short[d] = true;

        // compute llhw
        for (int i = 0; i < idx; i ++) {
            params[tn].ll += lgamma(alpha + tk[nidx[i]]) - lgamma_alpha;
            used[nidx[i]] = 0;
        }


        params[tn].ll -= lgamma(alpha * K + N) - lgamma(alpha * K);

        // traverse doc first time, accept word proposal
        for (int i = 0; i < N; i ++) {
            tt = local_topics[i];
            nk[tt] --;
            tk[tt] --;
            nk_new[tt] --;

            float b = tk[tt] + alpha;
            float d = nk[tt] + vbeta;

            for (unsigned m = 0; m < mhstep; m ++) {
                ttt = local_mhs[i * mhstep + m];
                float a = tk[ttt] + alpha;
                float c = nk[ttt] + vbeta;

                float ad = a * d;
                float bc = b * c;
                bool accept = params[tn].gen.Rand32() * bc < ad * std::numeric_limits<uint32_t>::max();
                float ac = ad / bc;
                if (ac > 1) ac = 1;
                one_over_accept_local += ac;
                sample_times_local += 1;

                if (accept) {
                    accept_times_local += 1;
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }

            nk[tt] ++;
            tk[tt] ++;
            local_topics[i] = tt;
            nk_new[tt] ++;
        }
        // traverse doc second time, generate doc proposal
        double new_topic = alpha * K / (alpha * K + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * new_topic;
        for (int i = 0; i < N; i ++) {
            topics[widx[di+i]] = local_topics[i];
            for (int m = 0; m < mhstep; m ++) {
                uint32_t si = widx[di + i] * mhstep + m;
                if (params[tn].gen.Rand32() < new_topic_th) {
                    mhs[si] = params[tn].gen.Rand32() % K;
                } else {
                    mhs[si] = local_topics[params[tn].gen.Rand32() % N];
                }
            }
        }

        params[tn].one_over_accept_sum += one_over_accept_local;
        params[tn].sample_times += sample_times_local;
        params[tn].accept_times += accept_times_local;
    }

    parallel_reduce_nk(global_nk, params, K);

    for (int i = 0; i < THREAD_NUM; i ++)
        global_ll += params[i].ll;

    double one_over_accept_all = 0;
    long sample_times_all = 0;
    long accept_times_all = 0;
    for (int i = 0; i < THREAD_NUM; i ++) {
        one_over_accept_all += params[i].one_over_accept_sum;
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }
    printf("one_over_accept_all=%f sample_times_all=%ld accept_times_all=%ld one_over_accept=%f accept_per=%f\n",
            one_over_accept_all, sample_times_all, accept_times_all,
            one_over_accept_all/sample_times_all,
            accept_times_all * 1.0 / sample_times_all);
    return global_ll;
}


double visit_by_col_parallel_separate(int D, int V, int K,
        int * lws, int * sws, int * global_nk,
        short * ltopics, short *stopics,
        short * mhs,
        float alpha, float beta, float vbeta,
        thread_params * params, int mhstep) {

    double lgamma_beta = lgamma(beta);
    double global_ll = 0.0;

    parallel_copy_nk(global_nk, params, K);

#pragma omp parallel for num_threads(THREAD_NUM)
    for (int w = 0; w < V; w ++) {
        int tn = omp_get_thread_num();
        int N = lws[w + 1] - lws[w];
        short tt, ttt, k;
        int idx;
        float p;

        if (N == 0)
            continue;

        int wi = lws[w];
        int * tk = params[tn].tk;
        int * nidx = params[tn].nidx;
        int * used = params[tn].used;
        int * nk = params[tn].nk;
        int * nk_new = params[tn].nk_new;
        short * local_topics = ltopics + wi;
        short * local_mhs    = mhs + wi * mhstep;
        double one_over_accept_local = 0.0;
        long sample_times_local = 0;
        long accept_times_local = 0;

        // compute wk for word on the fly
        memset(tk, 0, K * sizeof(int));
        
        // add wk from the long documents
        idx = 0;
        for (int i = 0; i < N; i ++) {
            k = local_topics[i];
            tk[k] ++;
            if (used[k] == 0) {
                nidx[idx ++] = k;
                used[k] = 1;
            }
        }
        
        // add wk from the short documents
        for (int i = sws[w]; i < sws[w + 1]; i ++) {
            k = stopics[i];
            tk[k] ++;
            if (used[k] == 0) {
                nidx[idx ++] = k;
                used[k] = 1;
            }
        }
            
        // compute llhw
        for (int i = 0; i < idx; i ++) {
            params[tn].ll += lgamma(beta + tk[nidx[i]]) - lgamma_beta;
            used[nidx[i]] = 0; 
        }

        // traverse w first time, accept doc proposal
        for (int i = 0; i < N; i ++) {
            tt = local_topics[i];
            //if (local_mhs[i * MH_STEPS] < 0)
            //    continue;

            tk[tt] --;
            nk[tt] --;
            nk_new[tt] --;
            
            float b = tk[tt]  + beta;
            float d = nk[tt]  + vbeta;
            
            for (unsigned m = 0; m < mhstep; m ++) {
                ttt = local_mhs[i * mhstep + m];
                float a = tk[ttt] + beta;
                float c = nk[ttt] + vbeta;
                bool accept = params[tn].gen.Rand32() * b* c < a * d * std::numeric_limits<uint32_t>::max();
                float ac = a * d / (b*c);
                if (ac > 1) ac = 1;
                one_over_accept_local +=  ac;
                sample_times_local += 1;

                if (accept) {
                    accept_times_local += 1;
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }
            
            tk[tt] ++;
            nk[tt] ++;
            nk_new[tt] ++;
            local_topics[i] = tt;
        }

        // traverse w second time, compute word proposal
        p = (K * beta) / (K * beta + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * p;
        for (int i = 0; i < N; i ++) {
            //if (local_mhs[i * MH_STEPS] < 0)
            //    continue;
            for (int m = 0; m < mhstep; m ++) {
                uint32_t si = i * mhstep + m;
                if (params[tn].gen.Rand32() < new_topic_th) {
                    local_mhs[si] = params[tn].gen.rand_int(K);
                } else {
                    local_mhs[si] = local_topics[params[tn].gen.Rand32() % N];
                }
            }
        }

        params[tn].one_over_accept_sum += one_over_accept_local;
        params[tn].sample_times += sample_times_local;
        params[tn].accept_times += accept_times_local;

    }

    parallel_reduce_nk(global_nk, params, K);
    
    for (int k = 0; k < K; k ++) {
        if (global_nk[k] > 0)
          global_ll += - lgamma(global_nk[k] + vbeta) + lgamma(vbeta);
    }
    
    for (int i = 0; i < THREAD_NUM; i ++)
        global_ll += params[i].ll;

    double one_over_accept_all = 0;
    long sample_times_all = 0;
    long accept_times_all = 0;
    for (int i = 0; i < THREAD_NUM; i ++) {
        one_over_accept_all += params[i].one_over_accept_sum;
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }
    printf("one_over_accept_all=%f sample_times_all=%ld accept_times_all=%ld one_over_accept=%f accept_per=%f\n",
            one_over_accept_all, sample_times_all, accept_times_all,
            one_over_accept_all/sample_times_all,
            accept_times_all * 1.0 / sample_times_all);

    return global_ll;
}


double visit_by_row_parallel_dynamic(int ld, int V, int K,
        int * ds, int * widx, int * global_nk,
        bool * is_short,
        short * topics, short * mhs, 
        float alpha, float beta, float vbeta,
        thread_params * params, int *mhsteps, int max_mh) {

    double lgamma_alpha = lgamma(alpha);
    double global_ll = 0;

    parallel_copy_nk(global_nk, params, K);

    int c_mh = mhsteps[0];
    int n_mh = mhsteps[1];
    
    // traverse each doc
#pragma omp parallel for num_threads(THREAD_NUM)
    for (int d = 0; d < ld; d ++) {
        int tn = omp_get_thread_num();
        int N = ds[d + 1] - ds[d];
        if (N == 0)
            continue;

        short tt, ttt, k;
        int idx;
        float p;


        int* tk = params[tn].tk;
        int* nidx = params[tn].nidx;
        int* used = params[tn].used;
        int* nk = params[tn].nk;
        int* nk_new = params[tn].nk_new;
        short* local_topics = params[tn].local_topics;
        short* local_mhs = params[tn].local_mhs;
        double one_over_accept_local = 0.0;
        long sample_times_local = 0;
        long accept_times_local = 0;

        // compute dk for doc on the fly
        memset(tk, 0, K * sizeof(int));
        //memset(used, 0, K * sizeof(int));
        
        idx = 0;
        int di = ds[d];
        for (int i = 0; i < N; i ++) {
            k = topics[widx[di + i]];
            local_topics[i] = k;
            tk[k] ++;
            if (used[k] == 0) {
                used[k] = 1;
                nidx[idx ++] = k;
            }

            for (int m = 0; m < c_mh; m ++)
                local_mhs[i * max_mh + m] = mhs[widx[di + i] * max_mh + m];
        }

        //if (idx < 64)
        //    is_short[d] = true;

        // compute llhw
        for (int i = 0; i < idx; i ++) {
            params[tn].ll += lgamma(alpha + tk[nidx[i]]) - lgamma_alpha;
            used[nidx[i]] = 0;
        }


        params[tn].ll -= lgamma(alpha * K + N) - lgamma(alpha * K);

        // traverse doc first time, accept word proposal
        for (int i = 0; i < N; i ++) {
            tt = local_topics[i];
            nk[tt] --;
            tk[tt] --;
            nk_new[tt] --;

            float b = tk[tt] + alpha;
            float d = nk[tt] + vbeta;

            for (unsigned m = 0; m < c_mh; m ++) {
                ttt = local_mhs[i * max_mh + m];
                float a = tk[ttt] + alpha;
                float c = nk[ttt] + vbeta;

                float ad = a * d;
                float bc = b * c;
                bool accept = params[tn].gen.Rand32() * bc < ad * std::numeric_limits<uint32_t>::max();
                float ac = ad / bc;
                if (ac > 1) ac = 1;
                one_over_accept_local += ac;
                sample_times_local += 1;

                if (accept) {
                    accept_times_local += 1;
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }

            nk[tt] ++;
            tk[tt] ++;
            local_topics[i] = tt;
            nk_new[tt] ++;
        }
        // traverse doc second time, generate doc proposal
        double new_topic = alpha * K / (alpha * K + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * new_topic;
        for (int i = 0; i < N; i ++) {
            topics[widx[di+i]] = local_topics[i];
            for (int m = 0; m < n_mh; m ++) {
                uint32_t si = widx[di + i] * max_mh + m;
                if (params[tn].gen.Rand32() < new_topic_th) {
                    mhs[si] = params[tn].gen.Rand32() % K;
                } else {
                    mhs[si] = local_topics[params[tn].gen.Rand32() % N];
                }
            }
        }

        params[tn].one_over_accept_sum += one_over_accept_local;
        params[tn].sample_times += sample_times_local;
        params[tn].accept_times += accept_times_local;
    }

    parallel_reduce_nk(global_nk, params, K);

    for (int i = 0; i < THREAD_NUM; i ++)
        global_ll += params[i].ll;

    double one_over_accept_all = 0;
    long sample_times_all = 0;
    long accept_times_all = 0;
    for (int i = 0; i < THREAD_NUM; i ++) {
        one_over_accept_all += params[i].one_over_accept_sum;
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }
    printf("sample_times_all=%ld accept_times_all=%ld accept_per=%f c_mh=%d n_mh=%d\n",
            sample_times_all, accept_times_all,
            accept_times_all * 1.0 / sample_times_all,
	    c_mh, n_mh);
    return global_ll;
}


double visit_by_col_parallel_dynamic(int D, int V, int K,
        int * lws, int * sws, int * global_nk,
        short * ltopics, short *stopics,
        short * mhs,
        float alpha, float beta, float vbeta,
        thread_params * params, int *mhsteps, int max_mh) {

    double lgamma_beta = lgamma(beta);
    double global_ll = 0.0;

    parallel_copy_nk(global_nk, params, K);

    int c_mh = mhsteps[0];
    int n_mh = mhsteps[1];

#pragma omp parallel for num_threads(THREAD_NUM)
    for (int w = 0; w < V; w ++) {
        int tn = omp_get_thread_num();
        int N = lws[w + 1] - lws[w];
        short tt, ttt, k;
        int idx;
        float p;

        if (N == 0)
            continue;

        int wi = lws[w];
        int * tk = params[tn].tk;
        int * nidx = params[tn].nidx;
        int * used = params[tn].used;
        int * nk = params[tn].nk;
        int * nk_new = params[tn].nk_new;
        short * local_topics = ltopics + wi;
        short * local_mhs    = mhs + wi * max_mh;
        double one_over_accept_local = 0.0;
        long sample_times_local = 0;
        long accept_times_local = 0;

        // compute wk for word on the fly
        memset(tk, 0, K * sizeof(int));
        
        // add wk from the long documents
        idx = 0;
        for (int i = 0; i < N; i ++) {
            k = local_topics[i];
            tk[k] ++;
            if (used[k] == 0) {
                nidx[idx ++] = k;
                used[k] = 1;
            }
        }
        
        // add wk from the short documents
        for (int i = sws[w]; i < sws[w + 1]; i ++) {
            k = stopics[i];
            tk[k] ++;
            if (used[k] == 0) {
                nidx[idx ++] = k;
                used[k] = 1;
            }
        }
            
        // compute llhw
        for (int i = 0; i < idx; i ++) {
            params[tn].ll += lgamma(beta + tk[nidx[i]]) - lgamma_beta;
            used[nidx[i]] = 0; 
        }

        // traverse w first time, accept doc proposal
        for (int i = 0; i < N; i ++) {
            tt = local_topics[i];

            tk[tt] --;
            nk[tt] --;
            nk_new[tt] --;
            
            float b = tk[tt]  + beta;
            float d = nk[tt]  + vbeta;
            
            for (unsigned m = 0; m < c_mh; m ++) {
                ttt = local_mhs[i * max_mh + m];
                float a = tk[ttt] + beta;
                float c = nk[ttt] + vbeta;
                bool accept = params[tn].gen.Rand32() * b* c < a * d * std::numeric_limits<uint32_t>::max();
                float ac = a * d / (b*c);
                if (ac > 1) ac = 1;
                one_over_accept_local +=  ac;
                sample_times_local += 1;

                if (accept) {
                    accept_times_local += 1;
                    tt = ttt;
                    b = a;
                    d = c;
                }
            }
            
            tk[tt] ++;
            nk[tt] ++;
            nk_new[tt] ++;
            local_topics[i] = tt;
        }

        // traverse w second time, compute word proposal
        p = (K * beta) / (K * beta + N);
        uint32_t new_topic_th = std::numeric_limits<uint32_t>::max() * p;
        for (int i = 0; i < N; i ++) {
            for (int m = 0; m < n_mh; m ++) {
                uint32_t si = i * max_mh + m;
                if (params[tn].gen.Rand32() < new_topic_th) {
                    local_mhs[si] = params[tn].gen.rand_int(K);
                } else {
                    local_mhs[si] = local_topics[params[tn].gen.Rand32() % N];
                }
            }
        }

        params[tn].one_over_accept_sum += one_over_accept_local;
        params[tn].sample_times += sample_times_local;
        params[tn].accept_times += accept_times_local;

    }

    parallel_reduce_nk(global_nk, params, K);
    
    for (int k = 0; k < K; k ++) {
        if (global_nk[k] > 0)
          global_ll += - lgamma(global_nk[k] + vbeta) + lgamma(vbeta);
    }
    
    for (int i = 0; i < THREAD_NUM; i ++)
        global_ll += params[i].ll;

    double one_over_accept_all = 0;
    long sample_times_all = 0;
    long accept_times_all = 0;
    for (int i = 0; i < THREAD_NUM; i ++) {
        one_over_accept_all += params[i].one_over_accept_sum;
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }
    printf("sample_times_all=%ld accept_times_all=%ld accept_per=%f c_mh=%d n_mh=%d\n",
            sample_times_all, accept_times_all,
            accept_times_all * 1.0 / sample_times_all,
	    c_mh, n_mh);

    return global_ll;
}




double train_for_one_iter_parallel(int iter, int D, int V, int K, int N,
        int * ws, int * ds, int * widx, int * nk,
        int * didx, bool * is_short,
        short * topics, short * mhs,
        float alpha, float beta, float vbeta,
        int max_len) {
    double ll = 0;

    long start, end, stt;
    long row_tt, col_tt, iter_tt;
    start = get_current_ms();
    stt = start;

    thread_params params[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i ++) {
        params[i].init(K, max_len);
        params[i].ll = 0;
    }

    start = get_current_ms();
    printf("start visit col\n");
    fflush(stdout);
    ll += visit_by_col_parallel(D, V, K, ws, nk,
            topics, mhs, alpha, beta, vbeta, params);
    
    printf("finish visit col\n");
    fflush(stdout);
    end = get_current_ms();
    col_tt = end - start;

    for (int i = 0; i < THREAD_NUM; i ++) {
        params[i].ll = 0;
        params[i].one_over_accept_sum = 0;
        params[i].sample_times = 0;
        params[i].accept_times = 0;
    }

    start = get_current_ms();
    printf("start visit row\n");
    fflush(stdout);
    ll += visit_by_row_parallel(D, V, K, ds, widx, nk, is_short,
            topics, mhs, alpha, beta, vbeta, params);
    printf("finish visit row\n");
    fflush(stdout);

    end = get_current_ms();
    row_tt = end - start;
    iter_tt = end - stt;

    printf("iter=%d iter_tt=%ld col_tt=%ld row_tt=%ld ll=%f %f M tokens/s\n", 
            iter, iter_tt, col_tt, row_tt, ll, ((double) N/1e6) / (iter_tt / 1e3));
    fflush(stdout);
    return ll;
}

double train_for_one_iter_parallel_sep(int iter, int D, int V, int K, int N,
        int ld, int ln, int sd, int sn,
        int * lws, int * lds, int * lwidx, int * nk,
        int * sws, short * stopics, short * ltopics,
        bool * is_short,
        short * mhs, int mhstep,
        float alpha, float beta, float vbeta,
        int max_len) {
    double ll = 0;

    long start, end, stt;
    long row_tt, col_tt, iter_tt;
    start = get_current_ms();
    stt = start;

    thread_params params[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i ++) {
        params[i].init(K, max_len, mhstep);
        params[i].ll = 0;
    }

    start = get_current_ms();
    printf("start visit col\n");
    fflush(stdout);
    ll += visit_by_col_parallel_separate(D, V, K, lws, sws, nk,
            ltopics, stopics,
            mhs, alpha, beta, vbeta, params, mhstep);
    
    printf("finish visit col\n");
    fflush(stdout);
    end = get_current_ms();
    col_tt = end - start;

    for (int i = 0; i < THREAD_NUM; i ++) {
        params[i].ll = 0;
        params[i].one_over_accept_sum = 0;
        params[i].sample_times = 0;
        params[i].accept_times = 0;
    }

    start = get_current_ms();
    printf("start visit row\n");
    fflush(stdout);
    ll += visit_by_row_parallel_seperate(ld, V, K, lds, lwidx, nk, is_short,
            ltopics, mhs, alpha, beta, vbeta, params, mhstep);
    printf("finish visit row\n");
    fflush(stdout);

    end = get_current_ms();
    row_tt = end - start;
    iter_tt = end - stt;

    printf("iter=%d iter_tt=%ld col_tt=%ld row_tt=%ld ll=%f %f M tokens/s\n", 
            iter, iter_tt, col_tt, row_tt, ll, ((double) N/1e6) / (iter_tt / 1e3));
    fflush(stdout);
    return ll;
}

double train_for_one_iter_parallel_dyn(int iter, int D, int V, int K, int N,
        int ld, int ln, int sd, int sn,
        int * lws, int * lds, int * lwidx, int * nk,
        int * sws, short * stopics, short * ltopics,
        bool * is_short,
        short * mhs, int * mhsteps,
        float alpha, float beta, float vbeta,
        int max_len, int max_mh) {
    double ll = 0;

    long start, end, stt;
    long row_tt, col_tt, iter_tt;
    start = get_current_ms();
    stt = start;

    thread_params params[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i ++) {
        params[i].init(K, max_len, max_mh);
        params[i].ll = 0;
    }

    start = get_current_ms();
    printf("start visit col\n");
    fflush(stdout);
    ll += visit_by_col_parallel_dynamic(D, V, K, lws, sws, nk,
            ltopics, stopics,
            mhs, alpha, beta, vbeta, params, mhsteps, max_mh);

    long sample_times_all = 0;
    long accept_times_all = 0;

    for (int i = 0; i < THREAD_NUM; i ++) {
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }

    double one_over_accept = sample_times_all * 1.0 / accept_times_all;
    int temp_mh = (int) one_over_accept + 1;
    int next_mh = std::min(std::max(mhsteps[0], temp_mh), max_mh);
    printf("next_mh=%d\n", next_mh);

    mhsteps[0] = mhsteps[1];
    mhsteps[1] = next_mh;
    
    printf("finish visit col\n");
    fflush(stdout);
    end = get_current_ms();
    col_tt = end - start;

    for (int i = 0; i < THREAD_NUM; i ++) {
        params[i].ll = 0;
        params[i].one_over_accept_sum = 0;
        params[i].sample_times = 0;
        params[i].accept_times = 0;
    }

    start = get_current_ms();
    printf("start visit row\n");
    fflush(stdout);
    ll += visit_by_row_parallel_dynamic(ld, V, K, lds, lwidx, nk, is_short,
            ltopics, mhs, alpha, beta, vbeta, params, mhsteps, max_mh);

    sample_times_all = 0;
    accept_times_all = 0;

    for (int i = 0; i < THREAD_NUM; i ++) {
        sample_times_all += params[i].sample_times;
        accept_times_all += params[i].accept_times;
    }

    one_over_accept = sample_times_all * 1.0 / accept_times_all;
    temp_mh = (int) one_over_accept + 1;
    next_mh = std::min(std::max(mhsteps[0], temp_mh), max_mh);
    printf("next_mh=%d\n", next_mh);

    mhsteps[0] = mhsteps[1];
    mhsteps[1] = next_mh;
 
    printf("finish visit row\n");
    fflush(stdout);

    end = get_current_ms();
    row_tt = end - start;
    iter_tt = end - stt;

    printf("iter=%d iter_tt=%ld col_tt=%ld row_tt=%ld ll=%f %f M tokens/s\n", 
            iter, iter_tt, col_tt, row_tt, ll, ((double) N/1e6) / (iter_tt / 1e3));
    fflush(stdout);
    return ll;
}



int get_max_dim(int * ds, int D) {
    int len;
    int max_len = 0;
    for (int d = 0; d < D; d ++) {
        len = ds[d + 1] - ds[d];
        if (len > max_len)
            max_len = len;
    }
    return max_len;
}


void train_warp_nips_parallel() {
    std::string path = "data/nips.train";
    int V = 12420;
    int K = 1024;
    float alpha = 50.0 / K;
    float beta  = 0.01F;
    float vbeta = V * beta;

    lda::warp_data * data = new lda::warp_data(path, V);

    printf("load finished\n");
    int * nk = new int[K];
    memset(nk, 0, K * sizeof(int));

    int max_len = get_max_dim(data->_ds, data->_D);
    thread_params params[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i ++)
        params[i].init(K, max_len);

    
    init_warp_parallel(data->_D, data->_V, K, 
            data->_ws, data->_ds, data->_widx, nk,
            data->_topics, data->_mhs, params);

    printf("init finished\n");
    
    printf("data->_N=%d data->_D=%d\n", data->_N, data->_D);
    int * didx = new int[data->_N];
    bool * is_short = new bool[data->_D];

    memset(didx, 0, data->_N * sizeof(int));
    printf("didx[0]=%d\n", didx[0]);
    memset(is_short, false, data->_D * sizeof(bool));

    for (uint32_t i = 0; i < 300; i ++) {
        train_for_one_iter_parallel(i, data->_D, data->_V, K, data->_N,
                data->_ws, data->_ds, data->_widx, nk,
                didx, is_short,
                data->_topics, data->_mhs,
                alpha, beta, vbeta, max_len);
    }

    delete data;
    delete [] nk;

}

void train_warp_tencent_parallel() {
    std::string path = "data/long_docs.shuffle";
    int V = 88916;
    int K = 1024;
    float alpha = 50.0 / K;
    float beta  = 0.01F;
    float vbeta = V * beta;

    lda::warp_data * data = new lda::warp_data(path, V);

    printf("load finished\n");
    int * nk = new int[K];
    memset(nk, 0, K * sizeof(int));

    int max_len = get_max_dim(data->_ds, data->_D);
    thread_params params[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i ++)
        params[i].init(K, max_len);

    
    init_warp_parallel(data->_D, data->_V, K, 
            data->_ws, data->_ds, data->_widx, nk,
            data->_topics, data->_mhs, params);

    printf("init finished\n");
    
    printf("data->_N=%d data->_D=%d\n", data->_N, data->_D);
    int * didx = new int[data->_N];
    bool * is_short = new bool[data->_D];

    memset(didx, 0, data->_N * sizeof(int));
    printf("didx[0]=%d\n", didx[0]);
    memset(is_short, false, data->_D * sizeof(bool));

    for (uint32_t i = 0; i < 300; i ++) {
        train_for_one_iter_parallel(i, data->_D, data->_V, K, data->_N,
                data->_ws, data->_ds, data->_widx, nk,
                didx, is_short,
                data->_topics, data->_mhs,
                alpha, beta, vbeta, max_len);
    }

    delete data;
    delete [] nk;
}




int get_max_dim(uint32_t * ds, int D) {
    int len;
    int max_len = 0;
    for (int d = 0; d < D; d ++) {
        len = ds[d + 1] - ds[d];
        if (len > max_len)
            max_len = len;
    }
    return max_len;
}

void train_warp_nips() {
    std::string path = "data/nips.train";
    int V = 12420;
    int K = 1024;
    float alpha = 50.0F/K;
    float beta  = 0.01;
    float vbeta = V * beta;

    lda::warp_data * data = new lda::warp_data(path, V);

    int * nk = new int[K];
    int * tk = new int[K];
    int * nidx = new int[K];
    int * used = new int[K];

    memset(nk, 0, K * sizeof(int));
    memset(used, 0, K * sizeof(int));
    
    xorshift gen;
    init_warp(data->_D, data->_V, K, 
            data->_ws, data->_ds, data->_widx, nk,
            data->_topics, data->_mhs, gen);

    int max_len = get_max_dim(data->_ds, data->_D);
    short * local_topics = new short[max_len];
    short * local_mhs    = new short[max_len * MH_STEPS];

    for (int i = 0; i < 300; i ++) {
        train_for_one_iter(i, data->_D, data->_V, K, data->_N,
                data->_ws, data->_ds, data->_widx, nk,
                tk, nidx, used,
                data->_topics, data->_mhs,
                alpha, beta, vbeta,
                max_len, local_topics, local_mhs);
    }

    delete data;
    delete [] nk;
}

JNIEXPORT jdouble JNICALL Java_lda_jni_Utils_warpOneIter
  (JNIEnv * env, jclass obj, 
   jint jiter, jint jD, jint jV, jint jK, jint jN, 
   jintArray jws, jintArray jds, jintArray jwidx, jintArray jnk, 
   jintArray jdidx,
   jbooleanArray jis_short,
   jshortArray jtopics, jshortArray jmhs, 
   jint jMH_STEPS, 
   jdouble jalpha, jdouble jbeta, jdouble jvbeta,
   jint maxlen) {
    jboolean is_copy;
    
    int* ws = (int*)env->GetPrimitiveArrayCritical(jws, &is_copy);
    int* ds = (int*)env->GetPrimitiveArrayCritical(jds, &is_copy);
    int* widx = (int*)env->GetPrimitiveArrayCritical(jwidx, &is_copy);
    int* nk = (int*)env->GetPrimitiveArrayCritical(jnk, &is_copy);
    int* didx = (int*)env->GetPrimitiveArrayCritical(jdidx, &is_copy);
    short* topics = (short*)env->GetPrimitiveArrayCritical(jtopics, &is_copy);
    short* mhs = (short*)env->GetPrimitiveArrayCritical(jmhs, &is_copy);
    bool* is_short = (bool*)env->GetPrimitiveArrayCritical(jis_short, &is_copy);
    
    double ll = train_for_one_iter_parallel(jiter, jD, jV, jK, jN,
            ws, ds, widx, nk,
            didx, is_short,
            topics, mhs,
            jalpha, jbeta, jvbeta,
            maxlen);

    env->ReleasePrimitiveArrayCritical(jws, ws, 0);
    env->ReleasePrimitiveArrayCritical(jds, ds, 0);
    env->ReleasePrimitiveArrayCritical(jwidx, widx, 0);
    env->ReleasePrimitiveArrayCritical(jnk, nk, 0);
    env->ReleasePrimitiveArrayCritical(jdidx, didx, 0);
    env->ReleasePrimitiveArrayCritical(jis_short, is_short, 0);
    env->ReleasePrimitiveArrayCritical(jtopics, topics, 0);
    env->ReleasePrimitiveArrayCritical(jmhs, mhs, 0);

    return ll;
  }


JNIEXPORT jdouble JNICALL Java_lda_jni_Utils_warpOneIterSep
  (JNIEnv * env, jclass obj, 
   jint jiter, jint jD, jint jV, jint jK, jint jN, 
   jint jld, jint jln, jint jsd, jint jsn,
   jintArray jlws, jintArray jlds, jintArray jlwidx, jintArray jnk, 
   jintArray jsws, jshortArray jstopics, jshortArray jltopics,
   jbooleanArray jis_short,
   jshortArray jmhs, jint jMH_STEPS, 
   jdouble jalpha, jdouble jbeta, jdouble jvbeta,
   jint maxlen) {
    jboolean is_copy;
    
    int* lws = (int*)env->GetPrimitiveArrayCritical(jlws, &is_copy);
    int* sws = (int*)env->GetPrimitiveArrayCritical(jsws, &is_copy);
    int* lds = (int*)env->GetPrimitiveArrayCritical(jlds, &is_copy);
    int* lwidx = (int*)env->GetPrimitiveArrayCritical(jlwidx, &is_copy);
    int* nk = (int*)env->GetPrimitiveArrayCritical(jnk, &is_copy);
    short* stopics = (short*)env->GetPrimitiveArrayCritical(jstopics, &is_copy);
    short* ltopics = (short*)env->GetPrimitiveArrayCritical(jltopics, &is_copy);
    short* mhs = (short*)env->GetPrimitiveArrayCritical(jmhs, &is_copy);
    bool* is_short = (bool*)env->GetPrimitiveArrayCritical(jis_short, &is_copy);
    
    double ll = train_for_one_iter_parallel_sep(jiter, jD, jV, jK, jN,
            jld, jln, jsd, jsn,
            lws, lds, lwidx, nk,
            sws, stopics, ltopics,
            is_short,
            mhs, jMH_STEPS,
            jalpha, jbeta, jvbeta,
            maxlen);

    env->ReleasePrimitiveArrayCritical(jlws, lws, 0);
    env->ReleasePrimitiveArrayCritical(jsws, sws, 0);
    env->ReleasePrimitiveArrayCritical(jlds, lds, 0);
    env->ReleasePrimitiveArrayCritical(jlwidx, lwidx, 0);
    env->ReleasePrimitiveArrayCritical(jnk, nk, 0);
    env->ReleasePrimitiveArrayCritical(jis_short, is_short, 0);
    env->ReleasePrimitiveArrayCritical(jstopics, stopics, 0);
    env->ReleasePrimitiveArrayCritical(jltopics, ltopics, 0);
    env->ReleasePrimitiveArrayCritical(jmhs, mhs, 0);

    return ll;
  }

JNIEXPORT jdouble JNICALL Java_lda_jni_Utils_warpOneIterDyn
  (JNIEnv * env, jclass obj, 
   jint jiter, jint jD, jint jV, jint jK, jint jN, 
   jint jld, jint jln, jint jsd, jint jsn,
   jintArray jlws, jintArray jlds, jintArray jlwidx, jintArray jnk, 
   jintArray jsws, jshortArray jstopics, jshortArray jltopics,
   jbooleanArray jis_short,
   jshortArray jmhs, jintArray jmhsteps, 
   jdouble jalpha, jdouble jbeta, jdouble jvbeta,
   jint maxlen, jint jmax_mh) {
    jboolean is_copy;
    
    int* lws = (int*)env->GetPrimitiveArrayCritical(jlws, &is_copy);
    int* sws = (int*)env->GetPrimitiveArrayCritical(jsws, &is_copy);
    int* lds = (int*)env->GetPrimitiveArrayCritical(jlds, &is_copy);
    int* lwidx = (int*)env->GetPrimitiveArrayCritical(jlwidx, &is_copy);
    int* nk = (int*)env->GetPrimitiveArrayCritical(jnk, &is_copy);
    short* stopics = (short*)env->GetPrimitiveArrayCritical(jstopics, &is_copy);
    short* ltopics = (short*)env->GetPrimitiveArrayCritical(jltopics, &is_copy);
    short* mhs = (short*)env->GetPrimitiveArrayCritical(jmhs, &is_copy);
    bool* is_short = (bool*)env->GetPrimitiveArrayCritical(jis_short, &is_copy);
    int* mhsteps = (int*)env->GetPrimitiveArrayCritical(jmhsteps, &is_copy);
    
    double ll = train_for_one_iter_parallel_dyn(jiter, jD, jV, jK, jN,
            jld, jln, jsd, jsn,
            lws, lds, lwidx, nk,
            sws, stopics, ltopics,
            is_short,
            mhs, mhsteps,
            jalpha, jbeta, jvbeta,
            maxlen, jmax_mh);

    env->ReleasePrimitiveArrayCritical(jlws, lws, 0);
    env->ReleasePrimitiveArrayCritical(jsws, sws, 0);
    env->ReleasePrimitiveArrayCritical(jlds, lds, 0);
    env->ReleasePrimitiveArrayCritical(jlwidx, lwidx, 0);
    env->ReleasePrimitiveArrayCritical(jnk, nk, 0);
    env->ReleasePrimitiveArrayCritical(jis_short, is_short, 0);
    env->ReleasePrimitiveArrayCritical(jstopics, stopics, 0);
    env->ReleasePrimitiveArrayCritical(jltopics, ltopics, 0);
    env->ReleasePrimitiveArrayCritical(jmhs, mhs, 0);
    env->ReleasePrimitiveArrayCritical(jmhsteps, mhsteps, 0);

    return ll;
  }

int main(int argc, char * argv[]) {

    //train_warp_nips();
    train_warp_tencent_parallel();
}


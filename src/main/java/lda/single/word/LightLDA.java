package lda.single.word;

import lda.parallel.Alias;
import lda.utils.Document;
import lda.utils.Utils;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Created by leleyu on 2016/9/4.
 */
public class LightLDA {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public int[] nk;

//  SparseDocTopic[] ndk;

  short[][] ndk;
  public int MH_STEPS = 2;

  public int[] ds;
  public int[] ws;
  public int[] docIds;
  public short[] topics;
  public int[] widx;

  public double lgammaBeta;
  public double lgammaAlpha;
  public double lgammaAlphaSum;

  public int taskNum;

  public int[] build_tt;

  SampleTask[] sampleTasks;
  Future<Double>[] futures;

  public ThreadPoolExecutor executor;

  boolean test = false;

  public LightLDA(int D, int V, int K, float alpha, float beta,
                  List<Document> docs, int taskNum) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta  = beta;
    this.vbeta = V * beta;

    buildMat(docs);

    nk = new int[K];
//    ndk = new SparseDocTopic[D];
    ndk = new short[D][K];

    for (int d = 0; d < D; d ++) {
//      ndk[d] = new SparseDocTopic(K, ds[d + 1] - ds[d]);
//      int len = ds[d + 1] - ds[d];
//      if (len < Byte.MAX_VALUE)
//        ndk[d] = new S2BTightTraverseMap(len);
//      else
//        ndk[d] = new S2STightTraverseMap(Math.min(K, len));
      ndk[d] = new short[K];
    }

    lgammaBeta = Gamma.logGamma(beta);
    lgammaAlpha = Gamma.logGamma(alpha);
    lgammaAlphaSum = Gamma.logGamma(alpha * K);

    sampleTasks = new SampleTask[taskNum];
    futures = new Future[taskNum];
    build_tt = new int[taskNum];

    for (int i = 0; i < taskNum; i ++) {
      sampleTasks[i] = new SampleTask(i);

    }

    this.taskNum = taskNum;

    executor = new ThreadPoolExecutor(16, 32, 60,
            TimeUnit.HOURS,
            new LinkedBlockingQueue<Runnable>());
  }

  public void init() {
    Random rand = new Random(System.currentTimeMillis());
    for (int w = 0; w < V; w ++)
      init(w, rand);
  }

  public void init(int wid, Random rand) {
    for (int j = ws[wid]; j < ws[wid + 1]; j ++) {
      int d = docIds[j];
      short tt = (short) rand.nextInt(K);
      topics[j] = tt;
//      topics[ds[d] + didx[j]] = tt;
      nk[tt] ++;
//      ndk[d].inc(tt);
      ndk[d][tt] ++;
    }
  }


  public void buildMat(List<Document> docs) {
    Document doc;
    int[] wcnt = new int[V];

    N = 0;

    // Count word and build doc start idx
    ds = new int[D + 1];
    ds[0] = 0;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      for (int w = 0; w < doc.wids.length; w ++) {
        wcnt[doc.wids[w]] ++;
      }

      N += doc.wids.length;
      ds[d + 1] = N;
    }

    docIds   = new int[N];
    topics = new short[N];
    widx   = new int[N];

    // Build word start idx
    ws = new int[V + 1];
    ws[0] = 0;
    for (int i = 0; i < V; i ++)
      ws[i + 1] = ws[i] + wcnt[i];

    // build doc to word reverse idx
    for (int d = D - 1; d >= 0; d --) {
      doc = docs.get(d);
      int wid;
      int di = ds[d];
      for (int w = 0; w < doc.wids.length; w ++) {
        wid = doc.wids[w];
        int pos = ws[wid] + (--wcnt[wid]);
        docIds[pos] = d;
        widx[di + w] = pos;
      }
    }
  }

  public void buildAlias(int wid, Alias alias, int[] wk) {
    float[] qw = alias.probability;
    float Qw = 0.0F;
    for (int k = 0; k < K; k ++) {
      qw[k] = (wk[k] + beta) / (nk[k] + vbeta);
      Qw += qw[k];
    }

    for (int k = 0; k < K; k ++)
      qw[k] /= Qw;

    alias.build(qw);
  }

  public double sample(int wid, int[] nk, int[] wk, int taskId) {
    Alias alias = new Alias(K);

    // compute wk
    Arrays.fill(wk, 0);
    for (int j = ws[wid]; j < ws[wid + 1]; j ++) {
      short tt = topics[j];
      wk[tt] ++;
    }

    long start = System.currentTimeMillis();
    buildAlias(wid, alias, wk);
    build_tt[taskId] += System.currentTimeMillis() - start;

    Random rand = new Random(System.currentTimeMillis());

    for (int j = ws[wid]; j < ws[wid + 1]; j ++) {
      int did = docIds[j];
      short tt = topics[j];

      short ttt = sampleOneToken(did, wk, tt, tt, alias, rand);

      if (ttt != tt) {
        topics[j] = ttt;
        nk[tt] --;
        nk[ttt] ++;

        wk[tt] --;
        wk[ttt] ++;

//        ndk[did].dec(tt);
//        ndk[did].inc(ttt);
        ndk[did][tt] --;
        ndk[did][ttt] ++;
      }
    }

    double ll = 0;

    if (test) {
      for (int k = 0; k < K; k++)
        if (wk[k] > 0) {
          ll += Gamma.logGamma(wk[k] + beta) - lgammaBeta;
        }
    }

    return ll;
  }

  public short sampleOneToken(int did, int[] wk, int old, int s, Alias alias, Random rand) {
    int t , w_t_cnt, w_s_cnt;
    int n_t, n_s;
    float n_td_alpha, n_sd_alpha;
    float n_tw_beta, n_sw_beta, n_t_beta_sum, n_s_beta_sum;
    float proposal_t, proposal_s;
    float nom, denom;
    float rejection, pi;
    int m;

//    SparseDocTopic dk = ndk[did];
    short[] dk = ndk[did];

    for (int i = 0; i < MH_STEPS; i ++) {
      // word proposal
      t = alias.next();

      if (t != s) {
        rejection = rand.nextFloat();

        w_t_cnt = wk[t];
        w_s_cnt = wk[s];
        n_t = nk[t];
        n_s = nk[s];


//        n_td_alpha = dk.get(t) + alpha;
//        n_sd_alpha = dk.get(s) + alpha;
        n_td_alpha = dk[t] + alpha;
        n_sd_alpha = dk[s] + alpha;

        n_tw_beta  = w_t_cnt + beta;
        n_sw_beta  = w_s_cnt + beta;
        n_t_beta_sum = n_t + vbeta;
        n_s_beta_sum = n_s + vbeta;

        if (s == old) {
          n_sd_alpha --;
          n_sw_beta --;
          n_s_beta_sum --;
        }

        if (t == old) {
          n_td_alpha --;
          n_tw_beta --;
          n_t_beta_sum --;
        }

        proposal_s = (w_s_cnt + beta) / (n_s + vbeta);
        proposal_t = (w_t_cnt + beta) / (n_t + vbeta);

        nom = n_td_alpha * n_tw_beta * n_s_beta_sum * proposal_s;
        denom = n_sd_alpha * n_sw_beta * n_t_beta_sum * proposal_t;

        pi = nom / denom;

        m = -(rejection < pi ? 1: 0);
        s = (t & m) | (s & ~m);
      }

      // doc proposal
//      int len = docs.get(did).length;
      int len = ds[did + 1] - ds[did];
      float n_td_or_alpha = rand.nextFloat() * (len + alpha * K);
      if (n_td_or_alpha < len) {
        int t_idx = (int) n_td_or_alpha;
//        t = docs.get(did).topics[t_idx];
//        t = topics[ds[did] + t_idx];
        t = topics[widx[ds[did] + t_idx]];
//        t = topics[ds[did] + t_idx];
      } else {
        t = rand.nextInt(K);
      }

      if (t != s) {
        rejection = rand.nextFloat();

        w_t_cnt = wk[t];
        w_s_cnt = wk[s];
        n_t = nk[t];
        n_s = nk[s];

//        n_td_alpha = dk.get(t) + alpha;
//        n_sd_alpha = dk.get(s) + alpha;
        n_td_alpha = dk[t] + alpha;
        n_sd_alpha = dk[s] + alpha;

        n_tw_beta = w_t_cnt + beta;
        n_t_beta_sum = n_t + vbeta;
        n_sw_beta = w_s_cnt + beta;
        n_s_beta_sum = n_s + vbeta;

        proposal_s = n_sd_alpha;
        proposal_t = n_td_alpha;

        if (s == old) {
          n_sd_alpha --;
          n_sw_beta --;
          n_s_beta_sum --;
        }

        if (t == old) {
          n_td_alpha --;
          n_tw_beta --;
          n_t_beta_sum --;
        }

        nom = n_td_alpha * n_tw_beta * n_s_beta_sum * proposal_s;
        denom = n_sd_alpha * n_sw_beta * n_t_beta_sum * proposal_t;

        pi = nom / denom;

        m = - (rejection < pi ? 1: 0);
        s = (t & m) | (s & ~m);
      }
    }
    return (short) s;
  }

  public double computeWordLLHSummary() {
    double ll = 0.0;
    ll += K * Gamma.logGamma(beta * V);
    for (int k = 0; k < K; k ++) {
      ll -= Gamma.logGamma(nk[k] + beta * V);
    }
    return ll;
  }

  public double computeDocLLH() {
    double ll = 0;
    for (int d = 0; d < D; d ++) {
      int len = ds[d + 1] - ds[d];
      ll += lgammaAlphaSum - Gamma.logGamma(alpha * K + len);
      for (int i = 0; i < K; i ++) {
        if (ndk[d][i] != 0)
          ll += Gamma.logGamma(alpha + ndk[d][i]) - lgammaAlpha;
      }
    }
    return  ll;
  }

  public double waitForFutures() {
    double sum = 0.0;
    try {
      for (int i = 0; i < taskNum; i ++)
        sum += futures[i].get();
    } catch (InterruptedException e) {
      e.printStackTrace();
    } catch (ExecutionException e) {
      e.printStackTrace();
    }
    return sum;
  }

  public void reduceNk() {
    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = sampleTasks[i].nk;
      for (int k = 0; k < K; k ++) {
        localNk[k] -= nk[k];
      }
    }

    for (int i = 0; i < taskNum; i ++) {
      int[] localNk = sampleTasks[i].nk;
      for (int k = 0; k < K; k ++)
        nk[k] += localNk[k];
    }
  }

  public void iteration(int it) {

//    long start = System.currentTimeMillis();
//    double ll = 0;
//    for (int w = 0; w < V; w ++) {
//      ll += sample(w, nk, wk);
//    }
//    long train_tt = System.currentTimeMillis() - start;

    if (it % 5 == 0)
      test = true;
    else
      test = false;

    long start = System.currentTimeMillis();
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w ++)
      queue.add(w);

    for (int i = 0; i < taskNum; i ++) {
      sampleTasks[i].setTaskQueue(queue, nk);
      build_tt[i] = 0;
      futures[i] = executor.submit(sampleTasks[i]);
    }

    double ll = waitForFutures();
    reduceNk();
    long train_tt = System.currentTimeMillis() - start;


    if (test) {
      ll += computeWordLLHSummary();
      ll += computeDocLLH();
    }

    System.out.format("it=%d ll=%f train_tt=%d built_tt=%d\n",
            it, ll, train_tt, build_tt[0]);
  }


  class SampleTask implements Callable<Double> {

    private ConcurrentLinkedQueue<Integer> queue;
    private int[] nk;
    private int[] wk;
    private int taskId;

    public SampleTask(int taskId) {
      nk = new int[K];
      wk = new int[K];
      this.taskId = taskId;
    }

    public void setTaskQueue(ConcurrentLinkedQueue<Integer> queue, int[] nk) {
      this.queue = queue;
      System.arraycopy(nk, 0, this.nk, 0, K);
    }

    @Override
    public Double call() throws Exception {
      Integer wid;

      double ll = 0;

      while ((wid = queue.poll()) != null) {
        ll += sample(wid, nk, wk, taskId);
      }

      return ll;
    }
  }

  public static LightLDA read(String[] argv) throws IOException {
    String path = argv[0];
    List<Document> docs = Utils.read(path);
    int M = docs.size();
    int V = Integer.parseInt(argv[1]);
    int K = Integer.parseInt(argv[2]);
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    int taskNum = Integer.parseInt(argv[3]);
    int iter = Integer.parseInt(argv[4]);

    System.out.format("D=%d V=%d K=%d alpha=%f beta=%f iter=%d\n",
            M, V, K, alpha, beta, iter);
    LightLDA lda = new LightLDA(M, V, K, alpha, beta, docs, taskNum);
    return lda;
  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int M = docs.size();
    int V = 12420;
    int K = 1024;
    float alpha = 50.0F / K;
    float beta  = 0.01F;
    LightLDA lda = new LightLDA(M, V, K, alpha, beta, docs, 4);
    System.out.println("init");
    lda.init();

    for (int i = 0; i < 100; i ++) {
      lda.iteration(i);
    }
  }
  public static void cmds(String[] argv) throws IOException {
    LightLDA lda = read(argv);
    int iter = Integer.parseInt(argv[4]);

    lda.init();
    for (int i = 0; i < iter; i ++)
      lda.iteration(i);

    lda.executor.shutdown();
  }


  public static void main(String[] argv) throws IOException {
    nips(argv);
//    cmds(argv);
  }
}

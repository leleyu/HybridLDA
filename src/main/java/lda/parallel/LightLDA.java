package lda.parallel;

import lda.utils.CsrMat;
import lda.utils.Document;
import lda.utils.Utils;
import org.apache.commons.math3.special.Gamma;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Created by leleyu on 2016/12/21.
 */
public class LightLDA {

  public int D, V, K, N;
  public float alpha, beta, vbeta;

  public CsrMat mat;
  public short[] topics;
  public int[] nk;

  public int taskNum;
  public int MH_STEPS = 2;

  public double lgammaBeta;
  public double lgammaAlpha;
  public double lgammaAlphaSum;

  public int[] docLens;
  public Alias[] alias;
  public int[][] nwk;

  SampleTask[] sampleTasks;
  BuildTask [] buildTasks;
  InitTask[] initTasks;
  Future<Double>[] futures;

  private boolean test = false;
  private int step = 20;

  public ThreadPoolExecutor executor;


  public LightLDA(int D, int V, int K, float alpha, float beta,
                  List<Document> docs, int taskNum) {

    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta  = beta;
    this.vbeta = V * beta;

    docLens = new int[D];
    buildMat(docs);

    lgammaBeta = Gamma.logGamma(beta);
    lgammaAlpha = Gamma.logGamma(alpha);
    lgammaAlphaSum = Gamma.logGamma(alpha * K);

    sampleTasks = new SampleTask[taskNum];
    buildTasks  = new BuildTask[taskNum];
    initTasks   = new InitTask[taskNum];
    futures = new Future[taskNum];
    for (int i = 0; i < taskNum; i ++) {
      sampleTasks[i] = new SampleTask();
      buildTasks[i] = new BuildTask();
      initTasks[i] = new InitTask();
    }

    this.taskNum = taskNum;

    executor = new ThreadPoolExecutor(16, 32, 60,
            TimeUnit.HOURS,
            new LinkedBlockingQueue<Runnable>());
  }

  public LightLDA(int D, int V, int K, float alpha, float beta,
                  List<Document> docs, int taskNum, int MH_STEPS) {
    this(D, V, K, alpha, beta, docs, taskNum);
    this.MH_STEPS = MH_STEPS;
  }

  public void buildMat(List<Document> docs) {
    N = 0;

    D = docs.size();

    int[] ds = new int[D + 1];

    ds[0] = 0;
    for (int d = 0; d < D; d ++) {
      N += docs.get(d).length;
      docLens[d] = docs.get(d).length;
      ds[d + 1] = N;
    }

    System.out.format("N=%d V=%d\n", N, V);

    int[] cols = new int[N];

    Document doc;
    for (int d = 0; d < D; d ++) {
      doc = docs.get(d);
      int sd = ds[d];
      int wid;
      for (int w = 0; w < doc.length; w ++) {
        wid = doc.wids[w];
        cols[sd + w] = wid;
      }
    }

    mat = new CsrMat(D, V, ds, cols);
  }

  public void check() {
    int[] nkt = new int[K];
  }

  public void init() {
    topics = new short[N];
    nk = new int[K];
    alias = new Alias[V];

    nwk = new int[V][];
    for (int w = 0; w < V; w ++)
      nwk[w] = new int[K];

    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int d = 0; d < D; d ++)
      queue.add(d);

    for (int i = 0; i < taskNum; i ++) {
      initTasks[i].setQueue(queue);
      futures[i] = executor.submit(initTasks[i]);
    }

    waitForFutures();
    reduceInitNk();
  }

  public void init(int d, Random rand, int[] nk) {

    int[] ds = mat.rs;
    int[] tokens = mat.cols;
    for (int j = ds[d]; j < ds[d + 1]; j ++) {
      int wid = tokens[j];

      short tt = (short) rand.nextInt(K);
      synchronized (nwk[wid]) {
        nwk[wid][tt]++;
      }
      nk[tt] ++;
      topics[j] = tt;
    }
  }

  public void buildAlias(int wid, int[] small, int[] large) {
    float[] qw = new float[K];
    float Qw = 0.0F;
    for (int k = 0; k < K; k ++) {
      qw[k] = (nwk[wid][k] + beta) / (nk[k] + vbeta);
      Qw += qw[k];
    }

    for (int k = 0; k < K; k ++)
      qw[k] /= Qw;

    alias[wid] = new Alias(K);
    alias[wid].build(qw, small, large);
  }

  public void buildAlias(int wid, Alias alias, int[] small, int[] large) {
    float[] qw = alias.probability;
    float Qw = 0.0F;
    for (int k = 0; k < K; k ++) {
      qw[k] = (nwk[wid][k] + beta) / (nk[k] + vbeta);
      Qw += qw[k];
    }

    for (int k = 0; k < K; k ++)
      qw[k] /= Qw;

    this.alias[wid].build(qw, small, large);

  }

  public void buildAlias() {

    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int w = 0; w < V; w ++)
      queue.add(w);

    for (int i = 0; i < taskNum; i ++) {
      buildTasks[i].setTaskQueue(queue);
      futures[i] = executor.submit(buildTasks[i]);
    }

    waitForFutures();
  }

  public void sample(int did, int[] nk, int[] dk) {

    Random rand = new Random(System.currentTimeMillis());

    int wid;
    short tt, ttt;
    Alias alias;
    int[] ds = mat.rs;
    int[] tokens = mat.cols;

    for (int j = ds[did]; j < ds[did + 1]; j ++) {
      wid = tokens[j];
      tt  = topics[j];
      alias = this.alias[wid];

      ttt = sampleOneToken(did, wid, tt, tt, alias, rand, dk, nwk[wid], nk);

      if (ttt != tt) {
        nwk[wid][tt] --;
        nwk[wid][ttt] ++;

        nk[tt] --;
        nk[ttt] ++;

        dk[tt] --;
        dk[ttt] ++;

        topics[j] = ttt;
      }
    }
  }

  public short sampleOneToken(int did, int wid, int old, int s, Alias alias, Random rand, int[] dk, int[] wk, int[] nk) {
    int t , w_t_cnt, w_s_cnt;
    int n_t, n_s;
    float n_td_alpha, n_sd_alpha;
    float n_tw_beta, n_sw_beta, n_t_beta_sum, n_s_beta_sum;
    float proposal_t, proposal_s;
    float nom, denom;
    float rejection, pi;
    int m;

    for (int i = 0; i < MH_STEPS; i ++) {
      // word proposal
      t = alias.next();

      if (t != s) {
        rejection = rand.nextFloat();

        w_t_cnt = wk[t];
        w_s_cnt = wk[s];
        n_t = nk[t];
        n_s = nk[s];


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
      int len = docLens[did];
      float n_td_or_alpha = rand.nextFloat() * (len + alpha * K);
      if (n_td_or_alpha < len) {
        int t_idx = (int) n_td_or_alpha;
//        t = docs.get(did).topics[t_idx];
        t = topics[mat.rs[did] + t_idx];
      } else {
        t = rand.nextInt(K);
      }

      if (t != s) {
        rejection = rand.nextFloat();

        w_t_cnt = wk[t];
        w_s_cnt = wk[s];
        n_t = nk[t];
        n_s = nk[s];

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

  public void iteration(int it) {

    if (it % 5 == 0)
      test = true;
    else
      test = false;

    long start = System.currentTimeMillis();
    buildAlias();
    long built_tt = System.currentTimeMillis() - start;

    start = System.currentTimeMillis();
    ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();
    for (int d = 0; d < D; d += step)
      queue.add(d);

    for (int i = 0; i < taskNum; i ++) {
      sampleTasks[i].setTaskQueue(queue, nk);
      futures[i] = executor.submit(sampleTasks[i]);
    }

    double ll = waitForFutures();
    long train_tt = System.currentTimeMillis() - start;

    if (test) {
      ll += computeWordLLH();
      ll += computeWordLLHSummary();
    }

    reduceNk();

    System.out.format("it=%d ll=%f build_tt=%d train_tt=%d\n",
            it, ll, built_tt, train_tt);

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

  public void reduceInitNk() {
    Arrays.fill(nk, 0);
    for (int i = 0; i < taskNum; i ++)
      for (int k = 0; k < K; k ++)
        nk[k] += initTasks[i].nk[k];
  }

  public double computeWordLLH() {
    double ll = 0;
    for (int w = 0; w < V; w ++) {
      ll += computeWordLLH(w);
    }
    return ll;
  }

  public double computeWordLLH(int wid) {
    int[] wk = nwk[wid];
    double ll = 0;
    for (int k = 0; k < K; k ++)
      if (wk[k] > 0)
        ll += Gamma.logGamma(wk[k] + beta) - lgammaBeta;
    return ll;
  }

  public double computeWordLLHSummary() {
    double ll = 0.0;
    ll += K * Gamma.logGamma(beta * V);
    for (int k = 0; k < K; k ++) {
      ll -= Gamma.logGamma(nk[k] + beta * V);
    }
    return ll;
  }

  class InitTask implements Callable<Double> {
    private ConcurrentLinkedQueue<Integer> queue;
    private int[] nk;

    public InitTask() {
      nk = new int[K];
    }

    public void setQueue(ConcurrentLinkedQueue<Integer> queue) {
      this.queue = queue;
    }

    @Override
    public Double call() throws Exception {
      Integer did;

      Random rand = new Random(System.currentTimeMillis());

      while ((did = queue.poll()) != null) {
        init(did, rand, nk);
      }

      return 0.0;
    }

  }

  class BuildTask implements Callable<Double> {
    private ConcurrentLinkedQueue<Integer> queue;
    private int[] small;
    private int[] large;

    public BuildTask() {
      small = new int[K];
      large = new int[K];
    }

    public void setTaskQueue(ConcurrentLinkedQueue<Integer> queue) {
      this.queue = queue;
    }

    @Override
    public Double call() throws Exception {
      Integer wid;



      while ((wid = queue.poll()) != null) {
        if (alias[wid] != null)
          buildAlias(wid, alias[wid], small, large);
        else
          buildAlias(wid, small, large);
      }
      return 0.0;
    }
  }


  class SampleTask implements Callable<Double> {

    private ConcurrentLinkedQueue<Integer> queue;
    private int[] nk;
    private int[] dk;

    public SampleTask() {
      nk = new int[K];
      dk = new int[K];
    }

    public void setTaskQueue(ConcurrentLinkedQueue<Integer> queue, int[] nk) {
      this.queue = queue;
      System.arraycopy(nk, 0, this.nk, 0, K);
    }

    @Override
    public Double call() throws Exception {
      Integer dstart;

      double ll = 0;

      while ((dstart = queue.poll()) != null) {

        for (int did = dstart; did < dstart + step; did++) {
          if (did == D)
            break;

          // Calculate dk
          Arrays.fill(dk, 0);
          for (int j = mat.rs[did]; j < mat.rs[did + 1]; j++) {
            dk[topics[j]]++;
          }
          sample(did, nk, dk);

          if (test) {
            ll += lgammaAlphaSum - Gamma.logGamma(alpha * K + docLens[did]);
            for (int i = 0; i < K; i++) {
              if (dk[i] != 0)
                ll += Gamma.logGamma(alpha + dk[i]) - lgammaAlpha;
            }
          }
        }
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
    int mh   = Integer.parseInt(argv[5]);

    System.out.format("D=%d V=%d K=%d alpha=%f beta=%f iter=%d mh=%d\n",
            M, V, K, alpha, beta, iter, mh);
    LightLDA lda = new LightLDA(M, V, K, alpha, beta, docs, taskNum, mh);
    return lda;
  }

  public static void cmds(String[] argv) throws IOException {
    LightLDA lda = read(argv);
    int iter = Integer.parseInt(argv[4]);

    lda.init();
    for (int i = 0; i < iter; i ++)
      lda.iteration(i);

    lda.executor.shutdown();
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
    lda.init();

    for (int i = 0; i < 100; i ++) {
      lda.iteration(i);
    }

    lda.executor.shutdown();
  }

  public static void main(String[] argv) throws IOException {
//    nips(argv);
    cmds(argv);
  }

}

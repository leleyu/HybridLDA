package lda.single.doc;

import lda.parallel.Alias;
import lda.utils.*;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by leleyu on 2016/9/2.
 */
public class LightLDA implements DocLDA {

//  private final static Log LOG = LogFactory.getLog(LightLDA.class);

  public int D, V, K;
  public float alpha, beta, vbeta;

  public int[][] nwk;
  public int[] nk;
  public SparseDocTopic[] ndk;

  public List<Document> docs;
  public Alias[] alias;

  public final int MH_STEPS = 2;

  public LightLDA(int D, int V, int K,
                  float alpha, float beta,
                  List<Document> docs) {
    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta = beta;
    this.vbeta = beta * V;
    this.docs = docs;
  }

  public void init() {
    this.nwk = new int[V][K];
    this.nk  = new int[K];
    this.ndk = new SparseDocTopic[D];

    this.alias = new Alias[V];

    for (int w = 0; w < V; w ++) {
      nwk[w] = new int[K];
    }

    int wid, tt;
    Random rand = new Random(System.currentTimeMillis());
    for (int d = 0; d < D; d ++) {
      Document doc = docs.get(d);
      doc.topics = new short[doc.wids.length];
      doc.docId = d;
      ndk[d] = new SparseDocTopic(K, Math.min(K, doc.wids.length));
      for (int w = 0; w < doc.wids.length; w ++) {
        wid = doc.wids[w];
        tt  = rand.nextInt(K);
        nwk[wid][tt] ++;
        nk[tt] ++;
        ndk[d].inc(tt);
        doc.topics[w] = (short) tt;
      }
    }
  }

  public void buildAliasTable() {
    for (int wid = 0; wid < V; wid ++) {
      buildAliasTable(wid);
    }
  }

  public void buildAliasTable(int wid) {
    float[] qw = new float[K];
    float Qw = 0.0F;
    for (int k = 0; k < K; k ++) {
      qw[k] = (nwk[wid][k] + beta) / (nk[k] + vbeta);
      Qw += qw[k];
    }

    for (int k = 0; k < K; k ++) {
      qw[k] /= Qw;
    }

    alias[wid] = new Alias(qw);
  }

  public void trainOneDoc(Document doc) {
    Random rand = new Random(System.currentTimeMillis());

    int wid, tt, did, ttt;
    Alias alias;

    did = doc.docId;

    for (int w = 0; w < doc.wids.length; w ++) {
      wid   = doc.wids[w];
      tt    = doc.topics[w];
      alias = this.alias[wid];

      ttt = sampleOneToken(did, wid, tt, tt, alias, rand);

      if (ttt != tt) {
        nwk[wid][tt] --;
        nwk[wid][ttt] ++;

        nk[tt] --;
        nk[ttt] ++;

        ndk[did].dec(tt);
        ndk[did].inc(ttt);

        doc.topics[w] = (short) ttt;
      }
    }
  }

  public int sampleOneToken(int did, int wid, int old, int s, Alias alias, Random rand) {
    int t , w_t_cnt, w_s_cnt;
    int n_t, n_s;
    float n_td_alpha, n_sd_alpha;
    float n_tw_beta, n_sw_beta, n_t_beta_sum, n_s_beta_sum;
    float proposal_t, proposal_s;
    float nom, denom;
    float rejection, pi;
    int m;

    SparseDocTopic dk = ndk[did];
    int[] wk = nwk[wid];

    for (int i = 0; i < MH_STEPS; i ++) {
      // word proposal
      t = alias.next();

      if (t != s) {
        rejection = rand.nextFloat();

        w_t_cnt = wk[t];
        w_s_cnt = wk[s];
        n_t = nk[t];
        n_s = nk[s];


        n_td_alpha = dk.get(t) + alpha;
        n_sd_alpha = dk.get(s) + alpha;

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
      int len = docs.get(did).wids.length;
      float n_td_or_alpha = rand.nextFloat() * (len + alpha * K);
      if (n_td_or_alpha < len) {
        int t_idx = (int) n_td_or_alpha;
        t = docs.get(did).topics[t_idx];
      } else {
        t = rand.nextInt(K);
      }

      if (t != s) {
        rejection = rand.nextFloat();

        w_t_cnt = wk[t];
        w_s_cnt = wk[s];
        n_t = nk[t];
        n_s = nk[s];

        n_td_alpha = dk.get(t) + alpha;
        n_sd_alpha = dk.get(s) + alpha;

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
    return s;
  }

  public void trainOneIteration(int iter) {
    long start, end;

    start = System.currentTimeMillis();
    buildAliasTable();
    end   = System.currentTimeMillis();

    long build_tt = end - start;

    start = System.currentTimeMillis();
    for (Document doc: docs) {
      trainOneDoc(doc);
    }
    end = System.currentTimeMillis();

    long train_tt = end - start;

    start = System.currentTimeMillis();
    double llhw = new Utils().loglilikehood2(alpha, beta, nwk, ndk, nk, K, V, docs);
    end = System.currentTimeMillis();

    long eval_tt = end - start;
    System.out.println(String.format("iter=%d train_tt=%d built_tt=%d eval_tt=%d llhw=%f",
            iter, train_tt, build_tt, eval_tt, llhw));
  }

  public void testOneIteration() {

  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int D = docs.size();
    int V = 12420;
    int K = 1024;
    float alpha = 50.0F / K;
    float beta  = 0.01F;

    LightLDA lda = new LightLDA(D, V, K, alpha, beta, docs);
    lda.init();

    for (int i = 0; i < 200; i ++) {
      lda.trainOneIteration(i);
      lda.testOneIteration();
    }
  }

  public static void main(String [] argv) throws IOException {
//    PropertyConfigurator.configure("conf/log4j.properties");
    nips(argv);
//    cmds(argv);
  }

}

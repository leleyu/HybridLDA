package lda.single.word;

import lda.sparse.S2STightTraverseMap;
import lda.utils.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by leleyu on 2016/9/4.
 */
public class AliasLDA implements WordLDA {

  private final static Log LOG = LogFactory.getLog(AliasLDA.class);

  public int D, V, K;
  public double alpha, beta, vbeta;

  public final int MH_STEPS = 2;

  public int[] nk;
  public S2STightTraverseMap[] ndk;

  public double[] p;
  public List<Document> docs;
  public TokensAndParams[] words;

  public AliasLDA(int D, int V, int K, double alpha, double beta, List<Document> docs) {

    this.D = D;
    this.V = V;
    this.K = K;
    this.alpha = alpha;
    this.beta = beta;
    this.vbeta = V * beta;
    this.docs = docs;
  }

  @Override
  public void init() {

    nk  = new int[K];
    ndk = new S2STightTraverseMap[D];

    p = new double[K];

    for (int d = 0; d < D; d ++) {
      ndk[d] = new S2STightTraverseMap(Math.min(K, docs.get(d).length));
    }

    words = Utils.buildTokensByWord(docs, K, V);

    int tt, did;
    Random rand = new Random(System.currentTimeMillis());
    TokensAndParams param;
    for (int w = 0; w < V; w ++) {
      param = words[w];
      if (param == null)
        continue;

      for (int d = 0; d < param.tokens.length; d ++) {
        did = param.tokens[d];
        tt  = rand.nextInt(K);
        param.topics[d] = (short) tt;
        param.wtrow[tt] ++;
        nk[tt] ++;
        ndk[did].inc(tt);
      }
    }
  }

  @Override
  public void trainOneIteration(int iter) {
    long start, end;

    AliasMethod alias = new AliasMethod(K);

    start = System.currentTimeMillis();
    for (TokensAndParams param: words)
      if (param != null)
        sampleOneWord(param, alias);

    end = System.currentTimeMillis();

    long train_tt = end - start;

    start = System.currentTimeMillis();
    double llhw = new Utils().loglilikehood2(alpha, beta, words, ndk, nk, K, V, docs);
    end = System.currentTimeMillis();

    long eval_tt = end - start;

    LOG.info(String.format("iter=%d train_tt=%d eval_tt=%d llhw=%f",
            iter, train_tt, eval_tt, llhw));
  }

  public void testOneIteration() {

  }

  public double buildAliasTable(int[] wk, AliasMethod alias) {
    double Qw = 0.0;
    double[] qw = alias.probability;

    for (int k = 0; k < K; k ++) {
      qw[k] = (wk[k] + beta) / (nk[k] + vbeta);
      Qw += qw[k];
    }

    for (int k = 0; k < K; k ++)
      qw[k] /= Qw;

    alias.build(qw);
    return Qw;
  }

  public void sampleOneWord(TokensAndParams param, AliasMethod alias) {
    double Qw = buildAliasTable(param.wtrow, alias);
    Random rand = new Random(System.currentTimeMillis());

    int tt, ttt, did;
    int kk, vv;
    double psum, u;

    S2STightTraverseMap dk;
    int[] wa = param.wtrow;

    for (int d = 0; d < param.tokens.length; d ++) {
      did = param.tokens[d];
      tt  = param.topics[d];
      dk  = ndk[did];

      wa[tt] --;
      nk[tt] --;
      dk.dec(tt);

      psum = 0.0;
      for (int i = 0; i < dk.size; i ++) {
//        kk = dk.key[dk.idx[i]];
        kk = dk.getKey(i);
        vv = dk.getVal(i);
//        vv = dk.value[dk.idx[i]];

        psum += vv * (wa[kk] + beta) / (nk[kk] + vbeta);
        p[i] = psum;
      }

      double select_pr = psum / (psum + alpha * Qw);

      ttt = -1;
      for (int r = 0; r < MH_STEPS; r ++) {

        if (rand.nextDouble() < select_pr) {
          u = rand.nextDouble() * psum;
//          ttt = dk.key[dk.idx[BinarySearch.binarySearch(p, u, 0, dk.size - 1)]];
          ttt = dk.getKey(BinarySearch.binarySearch(p, u, 0, dk.size - 1));
        } else {
          ttt = alias.next();
        }

        if (tt != ttt) {
          int s_n_dk = dk.get(tt);
          int t_n_dk = dk.get(ttt);

          double s_qw = alias.qw[tt];
          double t_qw = alias.qw[ttt];

          double temp_s = (wa[tt] + beta) / (nk[tt] + vbeta);
          double temp_t = (wa[ttt] + beta) / (nk[ttt] + vbeta);
          double acceptance = (t_n_dk + alpha) / (s_n_dk + alpha)
                  * temp_t / temp_s
                  * (s_n_dk * temp_s + alpha * s_qw * Qw)
                  / (t_n_dk * temp_t + alpha * t_qw * Qw);

          if (rand.nextDouble() < acceptance)
            tt = ttt;
        }
      }

      param.topics[d] = (short) ttt;
      wa[ttt] ++;
      nk[ttt] ++;
      dk.inc(ttt);
    }
  }

  public static void nips(String[] argv) throws IOException {
    String path = "nips.train";
    List<Document> docs = Utils.read(path);
    int D = docs.size();
    int V = 12420;
    int K = 1024;
    double alpha = 0.1;
    double beta  = 0.1;

    AliasLDA lda = new AliasLDA(D, V, K, alpha, beta, docs);
    lda.init();

    for (int i = 0; i < 200; i ++) {
      lda.trainOneIteration(i);
      lda.testOneIteration();
    }
  }

  public static void main(String[] argv) throws IOException {

    nips(argv);
  }
}

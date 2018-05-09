package lda.jni;

/**
 * Created by yulele on 16/9/11.
 */
public class Utils {

  public static native double warpOneIter(int iter, int D, int V, int K, int N,
                                          int[] ws, int[] ds, int[] widx, int[] nk,
                                          int[] didx, boolean[] is_short,
                                          short[] topics, short[] mhs, int MH_STEPS,
                                          double alpha, double beta, double vbeta,
                                          int maxLen);

  public static native double warpOneIterSep(int iter, int D, int V, int K, int N,
                                             int ld, int ln, int sd, int sn,
                                             int[] lws, int[] lds, int[] lwidx, int[] nk,
                                             int[] sws, short[] stopics, short[] ltopics,
                                             boolean[] is_short,
                                            short[] mhs, int MH_STEPS,
                                          double alpha, double beta, double vbeta,
                                          int maxLen);

  public static native double warpOneIterDyn(int iter, int D, int V, int K, int N,
                                             int ld, int ln, int sd, int sn,
                                             int[] lws, int[] lds, int[] lwidx, int[] nk,
                                             int[] sws, short[] stopics, short[] ltopics,
                                             boolean[] is_short,
                                             short[] mhs, int[] mhsteps,
                                             double alpha, double beta, double vbeta,
                                             int maxLen, int max_mh);

  public static native double warpInit(int D, int V, int K,
                                       int[] ws, int[] ds, int[] widx, int[] nk,
                                       short[] topics, short[] mhs, int MH_STEPS);


  public static native double visitByDoc(int iter, int D, int V, int K, int N,
                                         int[] ds, int[] widx, int[] nk,
                                         short[] topics, short[] mhs, int MH_STEPS,
                                         double alpha, double beta, double vbeta);

  public static native double visitByWord(int iter, int D, int V, int K, int N,
                                          int wid, int[] wk,
                                          int[] ws, int[] nk,
                                          short[] topics, short[] mhs, int MH_STEPS,
                                          double alpah, double beta, double vbeta);

}

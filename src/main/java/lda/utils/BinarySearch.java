package lda.utils;

/**
 * Created by leleyu on 2016/1/29.
 */
public class BinarySearch {

  /*! search the first x which x > u from [start,end] of p */
  public static int binarySearch(float[] p, float u, int start, int end) {
    int pstart = start, pend = end;

    while (pstart < pend) {
      if (pstart + 1 == pend) {
        if (p[pstart] > u)
          return pstart;
        else if (p[end] > u)
          return pend;
        else
          return -1;
      }

      int mid = (pstart + pend) / 2;
      double value = p[mid];
      if (value == u) {
        return mid + 1;
      }
      if (value < u) {
        pstart = mid + 1;
      } else {
        pend = mid;
      }
    }

    return pstart;
  }

  public static int binarySearch(double[] p, double u, int start, int end) {
    int pstart = start, pend = end;

    while (pstart < pend) {
      if (pstart + 1 == pend) {
        if (p[pstart] > u)
          return pstart;
        else if (p[end] > u)
          return pend;
        else
          return -1;
      }

      int mid = (pstart + pend) / 2;
      double value = p[mid];
      if (value == u) {
        return mid + 1;
      }
      if (value < u) {
        pstart = mid + 1;
      } else {
        pend = mid;
      }
    }

    return pstart;
  }
}

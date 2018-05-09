package lda.utils;

/**
 * Created by leleyu on 2016/9/28.
 */
public class CsrMat {
  public int row;
  public int col;
  public int[] rs;
  public int[] cols;
  public int[] vals;

  public CsrMat(int row, int col, int[] rs, int[] cols, int[] vals) {
    this.row = row;
    this.col = col;
    this.rs = rs;
    this.cols = cols;
    this.vals = vals;
  }

  public CsrMat(int row, int col, int[] rs, int[] cols) {
    this.row = row;
    this.col = col;
    this.rs = rs;
    this.cols = cols;
  }

  public long bytes() {
    long sum = 0;
    sum += rs.length * 4L;
    sum += cols.length * 4L;
    if (vals != null)
      sum += vals.length * 4L;
    return sum;
  }
}

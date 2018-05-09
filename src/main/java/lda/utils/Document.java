package lda.utils;

/**
 * Document class
 * Created by leleyu on 2015/12/23.
 */
public class Document {

  public int docId;
  public int length;
  public int[] wids;
  public short[] topics;

  public Document(int docId, int[] wids) {
    this.docId = docId;
    this.length = wids.length;
    this.wids = wids;
  }

  public int getLength() {
    return length;
  }

  public int getDocId() {
    return docId;
  }

  public int getWordId(int index) {
    return wids[index];
  }

  public void reset() {
    this.wids = null;
    this.topics = null;
  }
}

package lda.utils;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.fastutil.ints.IntComparator;

/**
 * Created by leleyu on 2016/3/7.
 */
public class TokensOneWord {

  private int wordId;

  private int[] docIds;
  private short[] topics;

  public TokensOneWord(int wordId, int[] docIds) {
    this.docIds = docIds;
    this.wordId = wordId;
    this.topics = new short[docIds.length];

//    IntArrays.quickSort(docIds, new IntComparator() {
//      @Override
//      public int compare(int i, int i1) {
//        return i - i1;
//      }
//
//      @Override
//      public int compare(Integer o1, Integer o2) {
//        return o1 - o2;
//      }
//    });
  }

  public TokensOneWord(int wordId, IntArrayList docIds) {
    this(wordId, docIds.toIntArray());
  }

  public int getWordId() {
    return wordId;
  }

  public int size() {
    return docIds.length;
  }

  public int getTopic(int index) {
    return topics[index];
  }

  public void setTopic(int index, int topic) {
    topics[index] = (short) topic;
  }

  public int getDocId(int index) {
    return docIds[index];
  }
}

package lda.parallel;

import it.unimi.dsi.fastutil.ints.IntArrayFIFOQueue;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Random;

/**
 * Created by leleyu on 2016/1/15.
 */
public class Alias {

  /* The random number generator used to sample from the distribution. */
  private final Random random;

  /* The probability and alias tables. */
  public int[] alias;
  public float[] probability;

  public float[] qw;

  public Alias(int len) {
    this.random = new Random(System.currentTimeMillis());
    this.alias  = new int[len];
    this.qw     = new float[len];
    this.probability = new float[len];
  }

  public Alias(float[] probabilities) {
    this(probabilities, new Random(System.currentTimeMillis()));
  }

  public Alias(float[] probabilities, Random random) {
    this.random = random;
    this.alias  = new int[probabilities.length];
    this.qw     = new float[probabilities.length];

    build(probabilities);
  }

  public void build(float[] p) {

    System.arraycopy(p, 0, qw, 0, qw.length);

    float avg = 1.0F / p.length;

    Deque<Integer> small = new ArrayDeque<>();
    Deque<Integer> large = new ArrayDeque<>();

    for (int i = 0; i < p.length; i ++) {
      if (p[i] >= avg) {
        large.add(i);
      }
      else {
        small.add(i);
      }
    }

    while (small.size() > 0 && large.size() > 0) {
      int less = small.pop();
      int more = large.pop();

      p[more] = p[more] + p[less] - avg;
      p[less] = p[less] * p.length;
      alias[less] = more;

      if (p[more] >= avg)
        large.add(more);
      else
        small.add(more);
    }

    while (small.size() > 0)
      p[small.pop()] = 1.0F;
    while (large.size() > 0)
      p[large.pop()] = 1.0F;

    probability = p;
  }

  public void build(float[] p, int[] small, int[] large) {
    System.arraycopy(p, 0, qw, 0, qw.length);

    float avg = 1.0F / p.length;

//    Deque<Integer> small = new ArrayDeque<>();
//    Deque<Integer> large = new ArrayDeque<>();

    int sp = 0, lp = 0;

    for (int i = 0; i < p.length; i ++) {
      if (p[i] >= avg) {
//        large.add(i);
        large[lp ++] = i;
      }
      else {
//        small.add(i);
        small[sp ++] = i;
      }
    }

//    while (small.size() > 0 && large.size() > 0) {
    while (sp > 0 && lp > 0) {
//      int less = small.pop();
//      int more = large.pop();
      int less = small[--sp];
      int more = large[--lp];

      p[more] = p[more] + p[less] - avg;
      p[less] = p[less] * p.length;
      alias[less] = more;

      if (p[more] >= avg)
//        large.add(more);
        large[lp ++] = more;
      else
        small[sp ++] = more;
//        small.add(more);
    }

    while (sp > 0)
      p[small[--sp]] = 1.0F;
//      p[small.pop()] = 1.0F;
    while (lp > 0)
      p[large[--lp]] = 1.0F;
//      p[large.pop()] = 1.0F;

    probability = p;
  }

  public int next() {
    /* Generate a fair die roll to determine which column to inspect. */
    int column = random.nextInt(probability.length);

    /* Generate a biased coin toss to determine which option to pick. */
    boolean coinToss = random.nextFloat() < probability[column];

    /* Based on the outcome, return either the column or its alias. */
    return coinToss ? column : alias[column];
  }

  public float[] getProbability() {
    return probability;
  }
}
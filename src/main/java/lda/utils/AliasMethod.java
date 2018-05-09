package lda.utils;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Random;

/**
 * Created by leleyu on 2016/1/15.
 */
public class AliasMethod {

  /* The random number generator used to sample from the distribution. */
  private final Random random;

  /* The probability and alias tables. */
  public int[] alias;
  public double[] probability;

  public double[] qw;
  public int sampleNum = 0;

  public AliasMethod(int len) {
    this.random = new Random(System.currentTimeMillis());
    this.alias  = new int[len];
    this.qw     = new double[len];
    this.probability = new double[len];
  }

  public AliasMethod(double[] probabilities) {
    this(probabilities, new Random(System.currentTimeMillis()));
  }

  public AliasMethod(double[] probabilities, Random random) {
    this.random = random;
    this.alias  = new int[probabilities.length];
    this.qw     = new double[probabilities.length];

    build(probabilities);
  }

  public void build(double[] p) {

    System.arraycopy(p, 0, qw, 0, qw.length);

    double avg = 1.0 / p.length;

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
      p[small.pop()] = 1.0;
    while (large.size() > 0)
      p[large.pop()] = 1.0;

    probability = p;
  }

  public int next() {
    /* Generate a fair die roll to determine which column to inspect. */
    int column = random.nextInt(probability.length);

    /* Generate a biased coin toss to determine which option to pick. */
    boolean coinToss = random.nextDouble() < probability[column];

    /* Based on the outcome, return either the column or its alias. */
    return coinToss ? column : alias[column];
  }

  public double[] getProbability() {
    return probability;
  }
}
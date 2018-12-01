package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 100;
    /** The t value */
    private static final int T = 11;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        String x = "[";
        String y = "[";
        String z = "[";
        for (int i = 1; i < 7; i++) {
        	System.out.println("Currently at " + i * 10000 + " iterations");
        	RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, i*10000);
            fit.train();
            System.out.println("RHC: " + ef.value(rhc.getOptimal()));
            x += ef.value(rhc.getOptimal()) + ", ";
            
            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            fit = new FixedIterationTrainer(sa, i * 10000);
            fit.train();
            System.out.println("SA: " + ef.value(sa.getOptimal()));
            y += ef.value(sa.getOptimal()) + ", ";
            
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 160, 20, gap);
            fit = new FixedIterationTrainer(ga, i*10000);
            fit.train();
            System.out.println("GA: " + ef.value(ga.getOptimal()));
            z += ef.value(ga.getOptimal()) + ", ";
        }
        System.out.println("x = " + x);
        System.out.println("y = " + y);
        System.out.println("z = " + z);
        
//        MIMIC mimic = new MIMIC(200, 20, pop);
//        fit = new FixedIterationTrainer(mimic, 1000);
//        fit.train();
//        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
    }
}

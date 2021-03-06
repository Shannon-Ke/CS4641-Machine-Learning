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
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesTest {
    /** The n value */
    static int N = 100;
    
    public static void main(String[] args) {
        String x = "[";
        String y = "[";
        String z = "[";
        for (int i = 1; i < 7; i++) {
        	N = i * 100;
        	int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new CountOnesEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges); 
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
            
        	RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 600);
            fit.train();
            //System.out.println(ef.value(rhc.getOptimal()));
            x += ef.value(rhc.getOptimal()) + ", ";
            SimulatedAnnealing sa = new SimulatedAnnealing(10000, .90, hcp);
            fit = new FixedIterationTrainer(sa, 600);
            fit.train();
            //System.out.println(ef.value(sa.getOptimal()));
            y += ef.value(sa.getOptimal()) + ", ";
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 3, 16, gap);
            fit = new FixedIterationTrainer(ga, 600);
            fit.train();
            //System.out.println(ef.value(ga.getOptimal()));
            z += ef.value(ga.getOptimal()) + ", ";
        }
        
        System.out.println("x = " + x);
        System.out.println("y = " + y);
        System.out.println("z = " + z);
        
//        MIMIC mimic = new MIMIC(50, 10, pop);
//        fit = new FixedIterationTrainer(mimic, 100);
//        fit.train();
//        System.out.println(ef.value(mimic.getOptimal()));
    }
}
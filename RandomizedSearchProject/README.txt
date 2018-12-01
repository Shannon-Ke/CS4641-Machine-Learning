Shannon Ke
CS 4641
Randomized Optimization Project

****************CARSTEST********************

My code is written in the CarsTest.java, where I run randomized hill climbing, simualted annealing, 
and the genetic algorithm on my processed-cars-data.csv. To get processed-cars-data.csv, I ran my 
cars-data.csv through the data_util.py which encoded the values. Since I was using Eclipse to run my code,
the place where the project is being run and my actual csv file is different, so in order to
run the code on the csv file, the correct path has to be input into FILENAME at line 36.

In Eclipse, I just ran the code by pressing the play button. Hyperparameters can be tuned for
RHC, SA, and GA on lines 193, 203, and 214. RHC just takes trainIterations, SA takes trainIterations, temperature,
and cooling rate, and GA takes trainIterations, population size, mating ratio, and mutation ratio.

The CarsTest.java file must be in the filepath ABAGAIL/src/opt/test folder once ABAGAIL has been cloned
from the ABAGAIL github repo:
	
	git clone https://github.com/pushkar/ABAGAIL.git

The CarsTest.java file is a modified version of mosdragon's PokerTest.java which can be found at :

	https://gist.github.com/mosdragon/53edf8e69fde531db69e?fbclid=IwAR1kVaf-iZUXqQMedwtdvbjRNtXHm21QWGQ_OIpenWmwwcLbv8QdejJjv10

The x, y, and z values were used so that I could easily transfer them to my plotting.py
file to plot them with matplotlib.

******************4-PEAKS PROBLEM******************

I used the 4-peaks test that came with the ABAGAIL library. Once you clone ABAGAIL, the FourPeaksTest.java file 
can be found in the filepath ABAGAIL/src/opt/test folder. Just hit run and fitness values for each 10000th iteration
will print out to the console. The x, y, and z values were used so that I could easily transfer them to my plotting.py
file to plot them with matplotlib.

*******************COUNT ONES PROBLEM*****************

Like the 4-peaks problem, I used the count ones test that came with ABAGAIL. It can be found in the same place as
the 4 peaks problem, and is named CountOnesTest.java. For the first test, I had the for loop changing the second
parameter in each call to FixedIterationTrainer, but the one in there now is when I was updating the size of the input
string.
Again, the x, y, and z values were used so that I could easily transfer them to my plotting.py
file to plot them with matplotlib.
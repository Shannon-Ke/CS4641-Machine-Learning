Shannon Ke
CS 4641
Reinforcement Learning README

I used Burlap and Maven to run the project in Eclipse. 

***********GRIDWORLD************

I got the gridworld code from juanjose49's github boiler plate code. The link:

https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4

Inside of his files, I altered Movement.java to chance stochastic values, BasicRewardFunction.java to
add my own reward matrices, and Easy and HardGRidWorldLauncher.java files to change up the walls
in the gridworld.

**********BLOCKDUDE************

I got the Block dude code from meredith-wenjunwu's github repository. The link: 

https://github.com/meredith-wenjunwu/CS-4641-Machine-Learning/tree/master/Assignment%204

I went ahead and used the code, only varying initial epsilon, learning rate, and Q-values when
prompted to in the console when running BlockDudeQL.java.

**********MAKING GRAPHS**********

Grid world and block dude came with visualizations, but for visualizing the data in the results folder 
(easyfridresults.txt, PI_results.txt, etc.), I used the plotting.py python file in Spyder to use 
matplotlib to graph Figures 5 and 6 and others in my paper.

The assignment4 folder should be inserted into src/main/java after cloning juanjose49's boiler plate code.
On a random note, isn't my large grid world cute? It was hard to make a nice maze while also keeping
it looking like a face though.
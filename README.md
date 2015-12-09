# neural-network-tutorial

![logo](images/logo.jpg)


## Introduction
 
I have been interested in artificial intelligence and artificial life for years and I read most of the popular books printed on the subject. I developed a grasp of most of the topics yet neural networks always seemed to elude me. Sure, I could explain their architecture but as to how they actually worked and how they were implemented… well that was a complete mystery to me, as much magic as science. I bought several books on the subject but every single one attacked the subject from a very mathematical and academic viewpoint and very few even gave any practical uses or examples. So for a long long time I scratched my head and hoped that one day I would be able to understand enough to experiment with them myself.
 
That day arrived some time later when - sat in a tent in the highlands of Scotland reading a book - I had a sudden blast of insight. It was one of those fantastic “eureka” moments and although Scotland is a beautiful place I couldn’t wait to get to a computer so I could try out what I’d just learnt. To my surprise the first neural net I programmed worked perfectly and I haven’t looked back since. I still have a great deal to learn, neural nets are a huge subject, but I hope I can share enough knowledge and enthusiasm to get you started on your own little projects. In many ways the fields of AI and A-Life are very exciting to work in. I think of these subjects as the explorers of old must have looked at all those vast empty spaces on the maps. There is so much to learn and discover.

I’ll start off by describing what a neural net actually is and what it’s architecture is, then I’ll do a little theory on how we get it to perform for us but I’ll try to use as little maths as possible. (Having some understanding of mathematics is impossible to avoid however and the deeper you get into this topic the more mathematics you are going to have to learn). Finally, we’ll get to the fun bit. I’ll come up with a little project I will program and take you through one step at a time. It will be in this last phase of the tutorial where I hope you get the same “eureka” feeling for neural nets as I did back in rainy old Scotland. Until then just sit back, absorb and be patient.


## So, what exactly is a Neural Network?

A neural network is mans crude way of trying to simulate the brain electronically. So to understand how a neural net works we first must have a look at how the old grey matter does its business…

Our brains are made up of about 100 billion tiny units called neurons. Each neuron is connected to thousands of other neurons and communicates with them via electrochemical signals. Signals coming into the neuron are received via junctions called synapses, these in turn are located at the end of branches of the neuron cell called dendrites. The neuron continuously receives signals from these inputs and then performs a little bit of magic. What the neuron does (this is over simplified I might add) is sum up the inputs to itself in some way and then, if the end result is greater than some threshold value, the neuron fires. It generates a voltage and outputs a signal along something called an axon. Don't worry too much about remembering all these new words as we won’t be using many of them from this moment onwards, just have a good look at the illustration and try to picture what is happening within this simple little cell.

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/biological_neuron.jpg">
</div>

Neural networks are made up of many artificial neurons. An artificial neuron is simply an electronically modelled biological neuron. How many neurons are used depends on the task at hand. It could be as few as three or as many as several thousand. One optimistic researcher has even hard wired 2 million neurons together in the hope he can come up with something as intelligent as a cat although most people in the AI community doubt he will be successful (Update: he wasn't!). There are many different ways of connecting  artificial neurons together to create a neural network but I shall be concentrating on the most common which is called a feedforward network. So, I guess you are asking yourself, what does an artificial neuron look like? Well here you go:

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/artificial_neuron.jpg">
</div>

Each input into the neuron has its own weight associated with it illustrated by the red circle. A weight is simply a floating point number and it's these we adjust when we eventually come to train the network. The weights in most neural nets can be both negative and positive, therefore providing excitory or inhibitory influences to each input. As each input enters the nucleus (blue circle) it's multiplied by its weight. The nucleus then sums all these new input values which gives us the activation (again a floating point number which can be negative or positive). If the activation is greater than a threshold value - lets use the number 1 as an example - the neuron outputs a signal. If the activation is less than 1 the neuron outputs zero. This is typically called a step function (take a peek at the following diagram and have a guess why).

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/neural_output.gif">
</div>


## Now for some maths

I now have to introduce you to some equations. I’m going to try to keep the maths down to an absolute minimum but it will be useful for you to learn some notation.  I’ll feed you the maths little by little and introduce new concepts when we get to the relevant sections. This way I hope your mind can absorb all the ideas a little more comfortably and you'll be able to see how the maths are put to work at each stage in the development of a neural net.

A neuron can have any number of inputs from one to n, where n is the total number of inputs. The inputs may be represented  therefore as x1, x2, x3… xn. And the corresponding weights for the inputs as w1, w2, w3… wn. Now, the summation of the weights multiplied by the inputs we talked about above can be written as x1w1 + x2w2 + x3w3 …. + xnwn,  which I hope you remember is the activation value. So…

a = x1w1+x2w2+x3w3... +xnwn

Fortunately there is a quick way of writing this down which uses the Greek capital letter sigma S, which is the symbol used by mathematicians to represent summation.


Maybe just to clarify what this means I should write it out in code. Assuming an array of inputs and weights are already initialized as x[n] and w[n] then:

double activation = 0;

for (int i=0; i<n; i++)
{
   activation += x[i] * w[i];
}

Got it? Now remember that if the activation > threshold we output a 1 and if activation < threshold we output a 0.

Let me illustrate everything I've shown you so far with a diagram.

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/artificial_neuron_math.jpg">
</div>

Please ensure that you understand exactly how to calculate the activation value before you move on.


## I understand all that but how do you actually use an artificial neuron?

Well, we have to link several of these neurons up in some way. One way of doing this is by organising the neurons into a design called a feedforward network. It gets its name from the way the neurons in each layer feed their output forward to the next layer until we get the final output from the neural network. This is what a very simple feedforward network looks like:

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/output_hidden_input.jpg">
</div>

Each input is sent to every neuron in the hidden layer and then each hidden layer’s neuron’s output is connected to every neuron in the next layer. There can be any number of hidden layers within a feedforward network but one is usually enough to suffice for most problems you will tackle. Also the number of neurons I've chosen for the above diagram was completely arbitrary. There can be any number of neurons in each layer, it all depends on the problem. By now you may be feeling a little dazed by all this information so I think the best thing I can do at this point would be to give you a real world example of how a neural net can be used in the hope that I can get your very own brain’s neurons firing!

You probably know already that a popular use for neural nets is character recognition. So let's design a neural network that will detect the number '4'. Given a panel made up of a grid of lights which can be either on or off, we want our neural net to let us know whenever it thinks it sees the character '4'. The panel is eight cells square and looks like this:

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/four_cell.jpg">
</div>

We would like to design a neural net that will accept the state of the panel as an input and will output either a 1 or zero. A 1 to indicate that it thinks the character ‘4’ is being displayed and 0 if it thinks it's not being displayed. Therefore the neural net will have 64 inputs, each one representing a particular cell in the panel and a hidden layer consisting of a number of neurons (more on this later) all feeding their output into just one neuron in the output layer. I hope you can picture this in your head because the thought of drawing all those little circles and lines for you is not a happy one <smile>.

Once the neural network has been created it needs to be trained. One way of doing this is initialize the neural net with random weights and then feed it a series of inputs which represent, in this example, the different panel configurations. For each configuration we check to see what its output is and adjust the weights accordingly so that whenever it sees something looking like a number 4 it outputs a 1 and for everything else it outputs a zero. This type of training is called supervised learning and the data we feed it is called a training set. There are many different ways of adjusting the weights, the most common for this type of problem is called backpropagation. I will not be going into backprop in this tutorial as I will be showing you a completely different way of training neural nets which requires no supervision whatsoever (and hardly any maths - woohoo!)

If you think about it, you could increase the outputs of this neural net to 10.  This way the network can be trained to recognize all the digits 0 through to 9. Increase them further and it could be trained to recognize the alphabet too!

Are you starting to get a feel for neural nets now? I hope so. But even if you’re not all that will hopefully change in a moment when you start to see some code.



## So, what’s our project going to be fup?

We are going to evolve virtual minesweepers to find and collect land-mines scattered about a very simple 2D world. ( In the original version of this tutorial the program evolved ants that collected food but I fancied a change. ;0) )

This is a screenshot of the application:

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/smart_sweepers.jpg">
</div>

As you can see it's a very simple display. The minesweepers are the things that look like tanks and the land-mines are represented by the green dots. Whenever a minesweeper finds a mine it is removed and another mine is randomly positioned somewhere else in the world, thereby ensuring there is always a constant amount of land-mines on display. The minesweepers drawn in red are the best performing minesweepers the program has evolved so far.

How is a neural net going to control the movement of a minesweeper? Well, just like the control of a real tank, the minesweepers are controlled by adjusting the speed of a left track and a right track. By applying various forces to the left and right side of a minesweeper we can give it a full range of movement. So the network requires two outputs, one to designate the speed of the left track, and the other to designate the speed of the right track.

The more thoughtful of you may be wondering how on earth we can apply varying forces when all we've discussed so far are binary networks outputting 1’s and 0’s. The secret to this is that instead of using a simple step (threshold) activation function we use one which softens the output of each neuron to produce a symmetrical curve. There are several functions which will do this and we are going to use one called the sigmoid function. (sigmoid, or sigmoidal is just a posh way of saying something is S shaped)


This equation may look intimidating to some of you but it’s very simple really. The e is a mathematical constant which approximates to 2.7183, the a is the activation into the neuron and p is a number which controls the shape of the curve. p is usually set to 1.0.

This function is terrific and comes in handy for all sorts of different uses because  it produces an output like this:

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/sigmoid.jpg">
</div>

The lower the value of p the more the curve begins to look like a step function. Also please note this curve is always centred around 0.5. Negative activation values produce a result less than 0.5, positive activation values produce a result greater than 0.5.

Therefore, to obtain a continuously graded output between 0 and 1 from our neurons we just have to put the sum of all the inputs x weights through the sigmoid function and Bob’s your uncle! So that’s our outputs dealt with, what about the inputs?

I have chosen to have four inputs. Two of them represent a vector pointing to the closest land-mine and the other two represent the direction the minesweeper is pointing. I call this vector, the minesweepers look-at vector.  These four inputs give the minesweeper's brain - its neural network - everything it needs to know to figure out how to orient itself towards the mines.

Now we have defined our inputs and our outputs what about the hidden layer/s? How do we decide how many layers we should have and how many neurons we should have in each layer? Well, this is a matter of guesswork and something you will develop a ‘feel’ for. There is no known rule of thumb although plenty of researchers have tried to come up with one. By default the simulation uses one hidden layer that contains six neurons although please spend some time experimenting with different numbers to see what effect they may have. I’d like to emphasise here that the more you play around with all the parameters the better the ‘feel’ you are going to develop and the better your neural networks will be.

Time to look at some code now. Here's a quick breakdown of the important classes.

CNeuralNet is the neural net class (surprise surprise).

CGenAlg is the genetic algorithm class.

CMineSweeper is a data and controller class for each minesweeper.

CController is the controller class which ties all the other classes together.

CParams is a class which loads in all the parameters for the application. They can be found in the file 'params.ini'. I strongly suggest you play around with the settings in this file when you start to play around with the code.

## The CNeuralNet class

Let’s get started on the neural network class, CNeuralNet. We want this class to be flexible so it can be used in other projects and as simple to use as possible. We need to be able to set up a neural network with any amount of inputs and outputs and any amount of neurons in any amount of hidden layers.  So how do we do this? Well, first we need to define structures for a neuron and a neuron layer. Let’s have a look at the definition of these structures… first the neuron:

struct SNeuron
{

   //the number of inputs into the neuron

   int m_NumInputs;



   //the weights for each input

   vector<double> m_vecWeight;



   //ctor

   SNeuron(int NumInputs);

};


This is very simple, we just need to keep a record of how many inputs there are into each neuron and a std::vector of doubles in which we will store all the weights. Remember, there's a weight for every input into the neuron. When a SNeuron object is created, all the weights are initialized with random values.

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="images/program_note.jpg">
</div>

This is the constructor for SNeuron:

SNeuron::SNeuron(int NumInputs): m_NumInputs(NumInputs+1)
{

  //we need an additional weight for the bias hence the +1

  for (int i=0; i<NumInputs+1; ++i)

  {

    //set up the weights with an initial random value

    m_vecWeight.push_back(RandomClamped());

  }

}


This takes the number of inputs going into the neuron as an argument and creates a vector of random weights. One weight for each input.

What’s that I hear you say? There’s an extra weight there! Well I’m glad you spotted that because that extra weight is quite important but to explain why it’s there I’m going to have to do some more maths. Remember that our activation was the sum of all the inputs x weights and that the output of the neuron was dependent upon whether or not this activation exceeded a threshold value (t)? And that this could be represented in equation form by

x1w1 + x2w2 + x3w3… + xnwn >= t

Because the network weights are going to be evolved it would be great if the threshold value could be evolved too. To make this easy I'm going to use a little trick to make it appear as a weight. All you have to do is subtract t from either side of the equation and we get:

x1w1 + x2w2 + x3w3… + xnwn – t >= 0

or we can write this another way:

x1w1 + x2w2 + x3w3… + xnwn + (-1)t >= 0

So you can see (hopefully) that we can treat the threshold as a weight that is always multiplied by an input of -1. This is usually referred to as the bias.

And that's why each neuron is initialized with one additional weight. Because now when the network is evolved we don’t have to worry about the threshold value as it's built in with the weights and will take care of itself. Good eh?

Lets get on with the rest of the neural net code… The next structure defines a layer of neurons.

struct SNeuronLayer
{

  //the number of neurons in this layer

  int m_NumNeurons;



  //the layer of neurons

  vector<SNeuron> m_vecNeurons;



  SNeuronLayer(int NumNeurons, int NumInputsPerNeuron);

};


As you can see this just groups together a bunch of neurons into a layer. The CNeuralNet class is much more exciting, so let's move on and take a look at its definition:

class CNeuralNet
{

private:


  int m_NumInputs;



  int m_NumOutputs;



  int m_NumHiddenLayers;



  int m_NeuronsPerHiddenLyr;



  //storage for each layer of neurons including the output layer

  vector<SNeuronLayer> m_vecLayers;



public:



  CNeuralNet();



  //have a guess... ;0)

  void CreateNet();



  //gets the weights from the NN

  vector<double> GetWeights()const;



  //returns the total number of weights in the net

  int GetNumberOfWeights()const;



  //replaces the weights with new ones

  void PutWeights(vector<double> &weights);



  //calculates the outputs from a set of inputs

  vector<double> Update(vector<double> &inputs);



  //sigmoid response curve

  inline double Sigmoid(double activation, double response);

};


Most of this should be self explanatory. The main work is done by the method Update. Here we pass in our inputs to the neural network as a std::vector of doubles and retrieve the output as another std::vector of doubles. This is really the only method we use after the CNeuralNetwork class has been initialized. We can just treat it as a black box, feeding it data and retrieving the output as if by magic. Let's take a closer look at this method:

vector<double> CNeuralNet::Update(vector<double> &inputs)
{

  //stores the resultant outputs from each layer

  vector<double> outputs;



  int cWeight = 0;



  //first check that we have the correct amount of inputs

  if (inputs.size() != m_NumInputs)

  {

    //just return an empty vector if incorrect.

    return outputs;

  }



  //For each layer....

  for (int i=0; i<m_NumHiddenLayers + 1; ++i)

  {

    if ( i > 0 )

    {

      inputs = outputs;

    }



    outputs.clear();



    cWeight = 0;



    //for each neuron sum the (inputs * corresponding weights).Throw

    //the total at our sigmoid function to get the output.

    for (int j=0; j<m_vecLayers[i].m_NumNeurons; ++j)

    {

      double netinput = 0;



      int NumInputs = m_vecLayers[i].m_vecNeurons[j].m_NumInputs;



      //for each weight

      for (int k=0; k<NumInputs - 1; ++k)

      {

        //sum the weights x inputs

        netinput += m_vecLayers[i].m_vecNeurons[j].m_vecWeight[k] *

                    inputs[cWeight++];

      }



      //add in the bias

      netinput += m_vecLayers[i].m_vecNeurons[j].m_vecWeight[NumInputs-1] *

                  CParams::dBias;



      //we can store the outputs from each layer as we generate them.

      //The combined activation is first filtered through the sigmoid

      //function

      outputs.push_back(Sigmoid(netinput, CParams::dActivationResponse));



      cWeight = 0;

    }

  }

  return outputs;

}


After this method has checked  the validity of the input vector it enters a loop which examines each layer in turn. For each layer, it steps through the neurons in that layer and sums all the inputs multiplied by the corresponding weights. The last weight added in for each neuron is the bias (remember the bias is simply a weight always tied to the value -1.0).  This value is then put through the sigmoid function to give that neurons output and then added to a vector which is fed back into the next iteration of the loop and so on until we have our output proper.

The other methods in CNeuralNet are used mainly by the genetic algorithm class to grab the weights from a network or to replace the weights of a network.



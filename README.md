# Musical notes procreation using Deep learning

![title](https://user-images.githubusercontent.com/66405302/162425568-ed0327cb-51ad-46af-8216-c4f79ff3368c.png)

Problem statement is to generate melody using deep learning. for that RNN and perticularly LSTM structure is used to predict the next sub-sequent notes. basically it is a time series problem wherein first we have to encode traditional musical notations (MIDI or abc notation) into numerical one and feed it into LSTM and subsequently predict the next notes which is in the numerical form. then after we have to decode the numerical values into respective notation either MIDI or abc notation. we can easily convert MIDI or abc notations into mp3 format and check the quality of the generated sound.


## Authors

- [@RonakPandya](https://www.github.com/RonakPandya072)
- [@AjayanSaroj]
- [@Amit]

## Introduction
साहित्यसङ्गीतकलाविहीनः साक्षात्पशुः पुच्छविषाणहीनः। 

तृणं न खादन्नपि जीवमानः तद्भागधेयं परमं पशूनाम्॥

A person destitute of literature, music or the arts is as good as an animal without a trail or horns. It is the good fortunes of the animals that he doesn't eat grass like them. (From Bhartuhari Nitishatakam Sloka 12)

Music is one of the most widely used signal streams. However, the cost and difficulty of tune creation is increasing, and increasingly humans are beginning to like the tune of small crowds, with the intention to motive tune not able to satisfy humans' needs.

Music technology the use of deep learning techniques has been a subject of interest for the past decades. Music proves to be a unique venture as compared to images, among three essential dimensions: Firstly, song is temporal, with a hierarchical structure with dependencies throughout time. Secondly, music includes more than one instruments which are interdependent and spread throughout time. Thirdly, song is grouped into chords, arpeggios and melodies — for this reason every time-step may also have more than one outputs. However, audio statistics has numerous properties that cause them to acquainted in a few methods to what's conventionally studied in deep learning (computer vision and natural language processing, or NLP). The sequential nature of the music reminds us of NLP, which we will use Recurrent Neural Networks for. There also are more than one ‘channels’ of audio (in phrases of tones, and gadgets), which are paying homage to images that Convolutional Neural Networks may be used for. Additionally, deep generative fashions are interesting new areas of research, with the ability to create practical artificial data. Some examples are Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs), in addition to language models in NLP.

## Dataset

Dataset
Dataset we have used in the project is from kern.humdrum.org which is open source dataset of virtual music scores in .krn data format. The dataset contains around 7,866,496 notes in 108,703 files. Dataset is also well classified according to composers and genres. To know more about each and every song and its origin, composer, type of the song etc. visit https://kern.humdrum.org/help/data/
![image1](https://user-images.githubusercontent.com/66405302/162423967-88a97f74-ab7f-4021-8f90-332217f98a01.png)


## Music theory concepts
An eighth note (American) or a quaver (British) is a musical note played for one eighth the duration of a whole note (semibreve), hence the name. This amounts to twice the value of the sixteenth note (semiquaver). It is half the duration of a quarter note (crotchet), one quarter the duration of a half note (minim), one eighth the duration of a whole note (semibreve), one-sixteenth the duration of a double whole note (breve), and one thirty-second the duration of a longa.

A whole note (American) or semibreve (British) in musical notation is a single note equivalent to or lasting as long as two half-notes or four quarter-notes.

![image2 png](https://user-images.githubusercontent.com/66405302/162424170-4fcfd5f8-fdf8-453b-9e00-af8844966e95.jpg)



## Library used

### Music21

Music21 is a Python-based toolkit for computer-aided musicology developed by MIT researchers.  
People use music21 to answer questions from musicology using computers, to study large datasets of music, to generate musical examples, to teach fundamentals of music theory, to edit musical notation, study music and the brain, and to compose music.
The 21 in music21 refers to its origins as a project nurtured at MIT. At MIT all courses have numbers and music, along with some other humanities departments, are numbered 21. The music departments of MIT, along with Harvard, Smith, and Mount Holyoke Colleges, helped bring this toolkit from its easiest roots to a mature system. Visit github page for documentation.

### midi2audio

Easily synthesize MIDI to audio or just play it.

It provides a Python and command-line interface to the FluidSynth synthesizer to make it easy to use and suitable for scripting and batch processing. In contrast, most MIDI processing software is GUI-based.


## Methodology

![pre_processing drawio](https://user-images.githubusercontent.com/66405302/162424768-077124e6-d4e3-4cc1-a962-61013ea89d8c.png)

### Stage 01: Data collection
As it is mentioned earlier, we have used kern.humdrum open source dataset. Dataset containing total of 12 folders of various composers’ songs. Mostly all songs are being originated from German. All the files are in the form of .krn which contains MIDI representation of the song. Follow the given link to download the dataset.
Dataset link: https://kern.humdrum.org/cgi-bin/browse?l=essen%2Feuropa%2Fdeutschl

### Stage 02: Data pre-processing 

It is the most crucial and time-consuming task. First we have to identify the .krn file from our dataset. However we cannot use all the .krn files from the dataset. We have to check if the given file is in the acceptable duration or not using acceptable_song() function defined in the colab notebook.

MIDI dataset contains two types of object, in the first part it includes Notes and Chords. Note objects contain information about the pitch, octave, and offset of the Note. Whereas in the second part it shows the duration (time) till which given node or pitch should sound.

In the music there are total 12 keys out of which is further classified as Major keys and minor keys. We can convert any major key to C# or minor key to a which gives ease of representation for our model.

![image3](https://user-images.githubusercontent.com/66405302/162425099-d0e757ea-845e-4468-81e3-08e4bb54f9bf.png)

We want to segregate the unique nodes from the MIDI files which afterwards can be used to encode the musical notation. Music21 provides inbuilt methods to segregate the nodes from the music files such as .getElementByClass() method. Encode_song() function is defined for encoding the notations and saved it in the json file as a dictionary format.

For example: given pitch = 60, duration= 1, acceptable_time = 0.25 can be represented as
[‘60’, ‘_’, ‘_’, ‘_’]

Now, data is ready to feed to the LSTM model. 

### Stage 03: Model Training and predicting

Our main problem is to predict next pair of melodies from a given seed value. Seed is given randomly by the users. But it should have some meaning. User can give as an input number as a seed that is defined in the dataset meaning that there is particular value associated with each node. For example, C# node has value 60, similarly all nodes (major or minor nodes) have defined values. You can open mapping.jason file to see the values.

Generating a next series of melody is not a complex problem as compare to text generation, filing the missing text, text summarization, text translation, speech to text, automatic image captioning etc. So, here we have designed simple LSTM model having 256 units and 38 outputs. The reason behind choosing 38 output is because we have 38 unique nodes in our dataset. A total of 311,846 parameters (weights and biases) is being trained in 20 epochs (It will take time around 0.5 hrs to train the model :) with accuracy over 90%.

A Melodygenerator class is defined in the code which is responsible to convert a given seed to one hot encoding and then it will load the trained model and predict the next melody and save it in .mid file with the help of Music21 library methods.

### Stage 04: .mid to .mp3 converter

.mp3 is popular music format for playing audio files. So, in this section with the help of sound-font file, .mid file is converted to .mp3 format. Sound-fonts are special files that can connect midi to mp3 format. It is comparable with word font files. With the help of audio2music python package .mid file is converted to .mp3 format. For playing the .mp3 file on jupyter/ colab environment IPython library’s Audio method is helpful. So, Play the music and enjoy the rhythm. Try to give different seed value to generate your own melody with the help of artificial intelligence. One can try with multiple instrument sounds and generate the instrumental song also. 

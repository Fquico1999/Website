---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "UBC Open Robotics"
summary: "Developing Software to compete in the RoboCup@Home competition."
authors: []
tags: ["Deep Learning", "Computer Vision"]
categories: []
date: 2020-06-05T15:18:22-07:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: true

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

_*The results for the Robocup@Home Education 2020 Online Competition are out! Check out our standing [below.](#result_2020)*_


_*[Click here](#contrib) if you want to skip to my involvement in this team.*_

## Overview
[UBC Open Robotics](http://www.ubcopenrobotics.ca/) is a student team comprised of 60 students split into three subteams - ArtBot, PianoBot, and Robocup@Home. I am a member of the software team in the RoboCup@Home subteam.

The objective of RoboCup@Home is to build a household assistant robot that can perform a variety of tasks, including carrying bags, introducing and seating guests at a party, answering a variety of trivia questions and more. Open Robotics is developing a robot to compete in the 2021 [RoboCup@Home Education Challenge](https://www.robocupathomeedu.org/) while in the meantime, our subteam will compete in the 2020 Competition using the Turtlebot 2 as our hardware platform.

## The Challenge
<a name="task1"></a>
The rules for the 2020 Challenge can be found [here](https://docs.google.com/document/d/1aNPdZDvf9X4HHF13eSge_eHDP9NmC6UDqeYDM9Xyjcg/edit), but they boil down to three specific tasks:
- Carry My Luggage - Navigation task
- Find My Mates - Vision task
- Receptionist - Speech task

#### Carry My Luggage

*Goal: The robot helps the operator to carry a bag to the car parked outside*

Starting at a predifined location, the robot has to find the operator and pick up the bag the operator is pointing to. After picking up the bag, the robot needs to indicate that it is ready to follow and then it must follow the operator while facing 4 obstacles along the way (crowd, small object, difficult to see 3D object, small blocked area).

#### Find My Mates

*Goal: The robot fetches the information of the party guests for the operator who knows only the names
of the guests.*
<a name="task3"></a> 
Knowing only the operator, the robot must identify unknown people and meet those that are waving. Afterwards, it must remember the person and provide a unique description of that person, as well as that person's location, to the operator.

#### Receptionist

*Goal: The robot has to take two arriving guests to the living room, introducing them to each other,
and offering the just-arrived guest an unoccupied place to sit.*
<a name="contrib"></a>
Knowing the host of the party, John, the robot must identify unknown guests, request their names and favourite drinks and then point to an empty seat where the guest can sit.

## My Contributions

My main contributions have been in speech recognition and in handle segmentation, targeting [task 3](#task3) and [task 1](#task1) respectively, however I also worked on facial recognition earlier in the project.

### Speech Recognition
*You can find this repository [here](https://github.com/UBC-OpenRobotics/SpeechRecognition)*

Speech recognition is implemented using [PocketSphinx](https://github.com/cmusphinx/pocketsphinx) which is based on [CMUSphinx](https://cmusphinx.github.io/). Which offers two modes of operation - Keyword Spotting (KWS) and Language Model (LM).

#### KWS

Keyword spotting tries to detect specific keywords or phrases, without imposing any type of grammer rules ontop.
Utilizing keyword spotting requires a .dic file and a .kwslist file.

The dictionary file is a basic text file that contains all the keywords and their phonetic pronunciation, for instance:

```
BACK	B AE K
FORWARD	F AO R W ER D
FULL	F UH L
```

These files can be generated [here](http://www.speech.cs.cmu.edu/tools/lextool.html) . 

The .kwslist file has each keyword and a certain threshold, more or less corresponding to the length of the word or phrase, as follows:

```
BACK /1e-9/
FORWARD /1e-25/
FULL SPEED /1e-20/
```

#### LM

Language model mode additionally imposes a grammer. To utilize this mode, .dic, .lm and .gram files are needed.

The dictionary file is the same as in KWS mode.

The .lm file can be generated, along with the .dic file, from a corpus of text, using [this tool](http://www.speech.cs.cmu.edu/tools/lmtool-new.html)

The `generate_corpus.py` script in `SpeechRecognition/asr/resources` sifts through the resource files from robocup's GPSRCmdGenerator and creates a corpus. The .dic and .lm files are generated from it by using the above tool.

Finally, the .gram file specifies the grammer to be imposed. For instance, if the commands we are expecting are always an action followed by an object or person and then a location, it might look like:

```
public <rule> = <actions> [<objects>] [<names>] [<locations>];

<actions> = MOVE | STOP | GET | GIVE

<objects> = BOWL | GLASS

<names> = JOE | JOEBOB

<locations> = KITCHEN | BEDROOM

```



### Handle Segmentation
*You can find this repository [here](https://github.com/UBC-OpenRobotics/HandleSegmentation)*

To be able to accurately pick up a bag, the robot must be able to detect where its handle is, as well as some information on how wide it is. To accomplish this, I trained a UNet model to segment images of handles.

UNet models are models that take as input an image and output a mask defining a region of interest. Producing data for these models requires labelling regions of interest on a variety of images. For that purpose I used two tools - [LableMe](http://labelme.csail.mit.edu/Release3.0/) or in [MakeSense.ai](https://www.makesense.ai/).

<figure>
    <img src='history.jpg' alt='training history' width="700"/>
    <figcaption>Training History for the Handle Segmentation Model</figcaption>
</figure> 

After training, model inference on the test set was promising.

<figure>
    <img src='handle_prediction_good.jpg' alt='Test set inference' width="800"/>
    <figcaption>Model Inference on Test Set: input image on the left, model prediction in the center and ground truth on the right</figcaption>
</figure> 

Additionally, some processing was done on the mask to obtain candidates for the apex of the handle, and its width. This allowed the model to output where the arm should grasp, like the sequence below. Additional work will be done to integrate the RGBD depth layer to obtain a depth location of the handle.

<a name="result_2020"></a>
{{< video library="1" src="openrobotics_handleseg.mp4" controls="" >}}


## 2020 RoboCup@Home Education Online Challenge

We (the software subteam) participated in the 2020 Online Challenge since it is the team's goal to develop our own hardware platform for 2021. Meanwhile, we put our software progress to the test on the Turtlebot2 platform.

Out of 8 finalists, we ended up in second place in the open category (meaning open hardware category), and first place in people's choice.

<figure>
    <img src='openrobotics_finals_open_plat.png' alt='open_category_finish' width="700"/>
</figure> 

<figure>
    <img src='openrobotics_finals_peoples_choice.png' alt='peoples_choice_finish' width="700"/>
</figure> 
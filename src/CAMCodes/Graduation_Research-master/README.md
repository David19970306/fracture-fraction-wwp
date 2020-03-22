<html>
<body>
<div>
      <img alt="er" src="https://github.com/Eljefemasao/Graduation_Research/blob/development/images_for_readme/gradcam.png" />
</div>

<div>    
<h1>A TABLE OF CONTENTS</h1>

  <ul>    
      <li><p><a href="#introduction">Introduction</a></p></li>
      <ul>
      <li><p><a href="#Research Background">Research Background</a></p></li>
      <li><p><a href="#What is Grad_CAM">What is Grad_CAM</a></p></li>
      </ul>
</ul>
</div>
<div>
<ul>
      <li><p><a href="#detect">Detect shadow region from an image applying Grad-CAM to binary shadow classifier CNN</a></p></li>
      <ul>
	<li><p><a href="#Shadow Detecter Architecture">Shadow Detecter Architecture</a></p></li>
	<li><p><a href="#Result applying our approach">Result applying our application at object image taken at outdoor</a></p></li>
      </ul>
</ul>
</div>

<div>
  <h1>Introduction of my Graduation Research</h1>
By conducting command which is described bottom, you can check my main code used at Graduation Research.
</div>  

<div>

 ```
  jupyter notebook classify_seesa_keras.ipynb
 ```
</div>

<h2>Research Background</h2>
<div>
<p>Generally, outdoor object are influenced by various optical phenomenons.
For instance, reflection of lights and shadows.
Those phenomenons makes edge and blobs and finally,they causes object appearance changes.
And, because of the object appearance changes effects a serious negative influence in outdoor object classifier CNNs,
      There are some needs to detecting shadow and removal from image.<br>
      So, this time, we suggest computional low-cost shadow detecter using Grad-CAM.
      By optimizing Grad-CMA to binary classifier Convolutional Neural Network which classifies shadow containing image, we believe it makes significant computional cost reduction. 
</p>
</div>

<h2>What is Grad_CAM</h2>
<div>
 Gradient-weighted Class Activation Mapping(Grad-CAM)is an excellent visualization idea for understanding Convolutional Neural Network functions. As more detail explanation of this technique, It uses the gradients of any target concept(say logits for 'dog' or even a caption),flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in an image for predicting the concept.
Furthermore, By piling up these localization map onto Guided Backpropagation output, it realizes high level visualization system.
  There are roughly two algorithm flows. One is the Class Activation Mapping(CAM) and the other one is Guided BackPropagation.
 CAM is one of the funduamental idea for Grad-CAM.
 <div>
 <img alt="er" src="https://github.com/Eljefemasao/Graduation_Research/blob/development/images_for_readme/gradcam_paper.png">  
 </div>
</div>
<h1>Detect shadow region from an image <br> applying Grad-CAM to binary shadow classifier CNN</h1>
<h2>Shadow Detecter Architecture</h2>

<div>
  Our new idea is to highlighting shadow region in an input image pixel by optimizing Grad-CAM idea.
  As shown bottom diagram,
</div>
<div>  
<img alt="er" src="https://github.com/Eljefemasao/Graduation_Research/blob/development/images_for_readme/binary.png" >
</div>


<h2>Result applying our approach at object image taken at outdoor</h2>
By applying that proposed method, you can detect shadow region in an image.
<div>
<img alt="er" src="https://github.com/Eljefemasao/Graduation_Research/blob/development/images_for_readme/gradcam.png" >
</div>

<h2>Requirements</h2>
You need to install bottom packages before conducting our code.
<h3>Major Dependencies</h3>
<ul>
<li>python==3.6.3</li>
<li>Jupyter==1.0.0</li>
<li>tensorboard==1.11.0</li>
<li>tensorflow-gpu==1.11.0</li>
<li>scikit-learn==0.19.1</li>
<li>keras==2.1.3</li>

</ul>
<h2>Files/Directories</h2>
<ul>
<li>classify_seesa_keras.ipynb: main code</li>
</ul>

</body>
</html>

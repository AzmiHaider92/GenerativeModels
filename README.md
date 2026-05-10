# GenerativeModels  

[1. Introduction](/Intro/README.md)

[2. Probabilistic Modeling](/ProbabilisticModeling/README.md)

[3. Autoregressive Models](/AutoRegressiveModels/README.md)

[4. Maximum Likelihood Learning](/MLLearning/README.md)

[5. Latent Variable Models](/LatentVariableModels/README.md)


5. Latent Variable Models
6. Learning Latent Variable Models
7. Normalizing Flows
8. Advanced Flow Models
9. Generative Adversarial Networks
10. Energy-Based Models
11. Score-Based Models
12. Diffusion Models
13. Advanced Diffusion Models
14. Evaluating Generative Models
15. Combining Generative Model Families
16. Discrete Deep Generative Models
17. Probabilistic Reasoning
  


















  
# 1. Autoregressive Model:    
Project description: DGM_x4.  
Solution: Autoregressive_Image_Modeling.ipynb   
    
   
Generally, autoregressive model over high-dimensional data $\mathbf{x}$ factor the joint distribution as the following product of conditionals:  
  
$$p(\mathbf{x})=p(x_1, ..., x_n)=\prod_{i=1}^{n} p(x_i|x_1,...,x_{i-1})$$     
  
![autoregressive_image_modeling](https://wiki.math.uwaterloo.ca/statwiki/images/thumb/5/5b/xi_img.png/500px-xi_img.png)  

  
Learning these conditionals is often much simpler than learning the joint distribution all together.  
Implementation of **PixelCNN** (Conditional Image Generation with PixelCNN Decoders; Oord, et al. 2016).  

Example of a probablistic image completion:    
![download (3)](https://github.com/user-attachments/assets/5a360286-7aeb-46c3-a17e-5e56a9fc0492)     

Result:
            
![666](https://github.com/user-attachments/assets/4e4c417b-6d6e-4b1d-8eaf-e23d54d21e9c)


# 3. Normalizing flows model:    
Project description: DGM_x6.  
Solution: NF.ipynb  

Implementation of NICE (Non-linear Independent Component Estimation; Dinh, et al. 2015).

Visualization of the process of normalizing flows from a normal distribution sample to a desired learned distribuion:  
![nf1](https://github.com/user-attachments/assets/a36b4fcd-ffdc-4883-9dd1-ecfb06841b1a)
![nf2](https://github.com/user-attachments/assets/81b9288b-dd3c-49d1-bc0c-4e4a12eed22e)

   
Visualization of traversing the latent space:  
![nf4](https://github.com/user-attachments/assets/ab838fa6-8fd5-44b4-bfc0-3a8715bab14b)


   

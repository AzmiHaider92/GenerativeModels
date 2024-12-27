# GenerativeModels  

Hello world,  
As part of the course "Deep generative models", I implemented 4 different generative models.    
I reccommend openning the notebook files (.ipynb) in google colab to see the already produced results and figures.

DGM_x5. VAE    
DGM_x7. DDPM    

Each has a notebook solution.  
  
  
# 1. Autoregressive Model:    
Project description: DGM_x4.  
Solution: Autoregressive_Image_Modeling.ipynb   
    
   
Generally, autoregressive model over high-dimensional data $\mathbf{x}$ factor the joint distribution as the following product of conditionals:  
  
$$p(\mathbf{x})=p(x_1, ..., x_n)=\prod_{i=1}^{n} p(x_i|x_1,...,x_{i-1})$$     
  
![autoregressive_image_modeling](https://wiki.math.uwaterloo.ca/statwiki/images/thumb/5/5b/xi_img.png/500px-xi_img.png)  

  
Learning these conditionals is often much simpler than learning the joint distribution all together.  
My implementation will focus on the **PixelCNN**.  

Example of a probablistic image completion:    
![download (3)](https://github.com/user-attachments/assets/5a360286-7aeb-46c3-a17e-5e56a9fc0492)     

Result:
            
![666](https://github.com/user-attachments/assets/4e4c417b-6d6e-4b1d-8eaf-e23d54d21e9c)


# 3. Normalizing flows model:    
Project description: DGM_x6.  
Solution: NF.ipynb  

Implementation of NICE (Non-linear Independent Component Estimation; Dinh, et al. 2015).

![nf3](https://github.com/user-attachments/assets/e525f58f-5778-401b-8655-023fdb59f553)
![nf4](https://github.com/user-attachments/assets/c48d4e2f-c9b8-4f16-a0b0-4c29d6b86707)
![nf1](https://github.com/user-attachments/assets/7f5cbd15-2cc0-454b-8a92-5d04712b4699)
![nf2](https://github.com/user-attachments/assets/4d01bfcc-bd75-41f0-b9b8-260f82ad6c63)


   

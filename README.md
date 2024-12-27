# GenerativeModels  

Hello world,  
As part of the course "Deep generative models", I implemented 4 different generative models.    

DGM_x5. VAE    
DGM_x6. NF    
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


# Normalizing flow model:   

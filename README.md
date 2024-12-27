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


![nf4](https://github.com/user-attachments/assets/ab838fa6-8fd5-44b4-bfc0-3a8715bab14b)
![nf1](https://github.com/user-attachments/assets/a36b4fcd-ffdc-4883-9dd1-ecfb06841b1a)
![nf2](https://github.com/user-attachments/assets/81b9288b-dd3c-49d1-bc0c-4e4a12eed22e)


   

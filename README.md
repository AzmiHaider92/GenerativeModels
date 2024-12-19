# GenerativeModels

Hello world,
As part of the course "Deep generative models", I implemented 4 different generative models.  
The description is under:
DGM_x4. Autoregressive  
DGM_x5. VAE  
DGM_x6. NF  
DGM_x7. DDPM  

Each has a notebook solution.


Autoregressice model:   
Generally, autoregressive model over high-dimensional data $\mathbf{x}$ factor the joint distribution as the following product of conditionals:  
$$p(\mathbf{x})=p(x_1, ..., x_n)=\prod_{i=1}^{n} p(x_i|x_1,...,x_{i-1})$$   
![autoregressive_image_modeling](https://github.com/user-attachments/assets/80d42bfa-cdf2-4139-bccf-0c39480be6c3)


Learning these conditionals is often much simpler than learning the joint distribution all together.
My implementation will focus on the **PixelCNN**.



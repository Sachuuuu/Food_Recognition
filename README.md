# Food_Recognition
  resize.py: Resize and store data set inresized_images directory
  create_pickle.py: Create data and target pickle files.
  trainer.py: Create and train the model. save weights in Food_V1.h5
  main.py : Run the model witn Test data
 
 # For Test the model
 Execute testapi.py - To start the server : http://127.0.0.1:5000/expression
 To send the POST request use postman, if you do not have it please download it from here
  
 Postman configurations:
 
 URL : http://127.0.0.1:5000/expression
 
 Body: {"text":"Base64 encoded image string"}
 
 Header: Content-Type:application/json
 
 # For Retrain the model
 Create an object from trainer.py and execute train method.
 
    traineObject = Trainer()
    
    traineObject.train()


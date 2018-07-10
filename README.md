We can use Googles TensorFlow library to help predict wither or not a customer is likely to be a return customer. We will need a listing of customer data from a customer database with attributes about their previous purchases, in a CSV file. 
 
Example of above mentioned file
Then using a Python program we can process to three data sets training (80%), validation (10%), and testing (10%). During the training of the model the validation set will be used to check the accuracy of the model. In this program there is an early stop function built in. This function is twofold; first is to prevent overfitting of the model and second is to ensure that the model will be the most accurate in the least amount of time. The test data will only be used at the end once the model has determined that it has reached peak accuracy

 
	We can see that the model as a test accuracy of 84.15% meaning that we can make informed decision on wither or not a customer would  worth investing time and money into. We can focus on the customer who are mostly likely to be a repeat customer. 	

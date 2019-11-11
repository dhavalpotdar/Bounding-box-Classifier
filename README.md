# Bounding-box-Classifier

A Deep Learning API to classifiy type of detected bounding boxes.

## Data
Data can include structured documents such as invoices, reports and bank statements. Labeling of the data needs to be done using an image
annotation tool such as [LabelImg](https://github.com/tzutalin/labelImg). This API can also be used at the end of table detection model
such as Faster RCNN in a bounding-box detection and classification pipeline, which is where the inspiration for this API came from. <br><br>
The data output by LabelImg should be in `YOLO` format, on which the `denormalize` method of the class can operate. If `xml` is chosen as the
output format, a separate function will need to be written to parse the files and extract bounding box coordinates. 

## Usage
```python
#### defining important parameters:

# define the size to which each crop of a bounding box is resized; the size
# should be square
resize_to=(224, 224)

# define the class mapping 
class_dict={0: 'Vertical Relational',
            1: 'Horizontal Entity',
            2: 'Multiline Text',
            3: 'Other Tabular',
            4: 'Header',
            5: 'Other',
            6: 'Logo'}

# define the subset of labels to be used in training (this feature is used
# in case of class imbalance or when very few examples of a particular class
# are found in the data)
use_labels=[0, 1, 2, 3, 5, 6]

# NOTE: the above parameters are default for read_train_data()

#### main script:
cc = ContainerClassifier()
cc.read_train_data(path='./annotated_data', resize_to=resize_to, 
                                            class_dict=class_dict, 
                                            use_labels=use_labels)

cc.train(model='resnet101', epochs=30, batch_size=4)
cc.save_model(name='resnet101_nopretrained_30epochs_224x224')

# uncomment if loading previously trained model:
# cc.load_model('./models/container_classifier_v1_0_0.json',
#                 './models/container_classifier_v1_0_0.h5')

# testing
cc.read_test_data(path='./test', resize_to=(224, 224))
cc.test_model(output_dir='test_output')  

```
## Output


### Training Description
- Training data: 1020 <br>
- Arch: resnet101 <br>
- Image Size: 224x224 <br>
- Pretrained: False <br>
- Epochs: 30 <br><br>


### Testing Metrics 


```python
Classification Report:

			      	      precision    recall  f1-score   support

	vertical relational   0            0.91      1.00      0.95        10
	horizontal entity     1            0.89      0.89      0.89         9
	multiline text	      2            1.00      0.81      0.90        16
	other tabular	      3            0.00      0.00      0.00         1
	other		      4            0.56      1.00      0.71         5
	logo	              5            1.00      0.67      0.80         3


                             accuracy                          0.86        44
                            macro avg      0.73      0.73      0.71        44
                         weighted avg      0.88      0.86      0.86        44


Confusion Matrix:

[[10  0  0  0  0  0]       0: vertical relational
 [ 0  8  0  0  1  0]	   1: horizontal entity	
 [ 1  0 13  0  2  0]	   2: multiline text
 [ 0  1  0  0  0  0]	   3: other tabular
 [ 0  0  0  0  5  0]	   4: other
 [ 0  0  0  0  1  2]]	   5: logo
 
 ```
 
 ### Sample Output Images
 1. Vertical Relational Table: <br><br>
 ![sample1](./sample_output_1.jpg) <br><br>
 
 2. Multiline Text: <br><br>
 ![sample2](./sample_output_2.jpg) <br><br>
 
 3. Horizontal Entity Table: <br><br>
 ![sample3](./sample_output_5.jpg) <br><br>
 
 4. Other:<br><br>
 ![sample4](./sample_output_4.jpg) <br><br>
 
 
 ## References
 The Taxonomy to classify table types was inspired by the following paper:<br>
 - Kyosuke Nishida, Kugatsu Sadamitsu, Ryuichiro Higashinaka, Yoshihiro Matsuo - Understanding the Semantic Structures of Tables with a
Hybrid Deep Neural Network Architecture - *Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17)* [arXiv](https://arxiv.org/pdf/1901.04672)
 
 ## License
 The contents of this repo are covered under the [MIT License]('./LICENSE').

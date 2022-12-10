# Evaluation of shape complexity based on view similarity
Automating the evaluation of shape complexity based on view similarity. Automating the process mentioned in this research paper [A view similarity-based shape complexity metric to guide part selection for additive manufacturing](https://www.researchgate.net/publication/364303199_A_view_similarity-based_shape_complexity_metric_to_guide_part_selection_for_additive_manufacturing)

## Summmary
* This is the backend application
* Developed using FastAPI
* It has an API that take input an stl file and Capture 22 external images from the fixed viewpoints
* It has an API that take input an axis and slice_height and Slice the stl cad model on given slice height continously
* It has an API that calculate the dissimilarity between the captured images of both external and internal and return excel file containing matrix
* It has an API that return final calculated shape complexity

## Requirement to run this application:
* Docker desktop should be installed and running in your machine

## To run this project in your local machine
```
  cd shape-complexity-backend
  docker-compose build
  docker-compose up -d
```

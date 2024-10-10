# sam-dino-api
Small REST API wrapping Metas Segment Anything Model and Grounding Dino for prompt based image segmentation

## Example
Can be called on images uploaded to AWS S3 using curl, e.g.

```
curl --header "Content-Type: application/json" --request POST --data '{"prompt":"grey pot,leafs", "image_url":"https://boum-data.s3.amazonaws.com/segmentation/Week_3_plant_9_40_0.jpg"}' http://3.253.15.150/segment_by_prompt
```

Resulting in the following segmentation

![Segmentation result](oeqmpgnj.jpg?raw=true "Segmentation result")

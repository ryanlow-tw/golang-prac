package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
)

func main() {
	
	rawData, err := base.ParseCSVToInstances("iris.csv", true)
	if err != nil {
		panic(err)
	}

	model := ensemble.NewRandomForest(500, 3)

	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	model.Fit(trainData)

	predictions, err := model.Predict(testData)
	if err != nil {
		panic(err)
	}

	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

}
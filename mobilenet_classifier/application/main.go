package main

import (
	"fmt"
	"os"
	"io/ioutil"
	"github.com/StefanCepa/mobilenet_classifier/utils"
)

func main(){


	model, err := ioutil.ReadFile(utils.ModelPath)

	if err != nil {
		fmt.Printf("[ERROR]: Reading MobileNet Protobuf File\n" + err.Error())
		os.Exit(1)
	}

	graph := utils.CreateGraphAndImportExistingModel(model)
		
	session := utils.CreateAndInitializeNewSession(graph)

	defer session.Close()

	input_tensor, err := utils.ImageToTensor(os.Args[1])

	if err != nil {
		fmt.Printf("[ERROR]: Converting Image to Tensor\n" + err.Error())
		os.Exit(1)
	}

	result := utils.Inference(session, graph, "input", "MobilenetV2/Predictions/Reshape_1", input_tensor)

	probabilities := result[0].Value().([][]float32)[0]

	utils.PrintBestLabel(probabilities, utils.LabelPath)	
}


package utils

import (
	"bufio"
	"log"
	"fmt"
	"os"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

/**
Arguments:
	- session *tf.Session: session object in which computational graph
			      that represents our pretrained MobileNet will
			      be executed
	- graph *tf.Graph: computational graph that represents our pretrained
			  MobileNet architecture.
	- input_node string: name of the input node of the computational graph
	- output_node string: name of the output node of the computational graph
	- input_tensor *tf.Tensor: tensor representation of the image which we
				  want to classify.

Return Value:
	- result []*tf.Tensor: a list of class probabilities
**/
func Inference(session *tf.Session, graph *tf.Graph, input_node string, output_node string, input_tensor *tf.Tensor) []*tf.Tensor{

	result, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation(input_node).Output(0): input_tensor,
		},
		[]tf.Output{
			graph.Operation(output_node).Output(0),		
		},
		nil,
	)

	if err != nil {
		fmt.Printf("[ERROR]: Inference\n" + err.Error())
		os.Exit(1)
	}

	return result
}


/**
Prints most likely class.

Arguments:
	- probabilities []float32: class probabilities
	- labelsFile string: path to the file which contains class labels

**/
func PrintBestLabel(probabilities []float32, labelsFile string) {
	bestIdx := 0
	for i, p := range probabilities {
		if p > probabilities[bestIdx] {
			bestIdx = i
		}
	}
	
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
	fmt.Printf("%2.0f%%  %s\n", probabilities[bestIdx]*100.0, labels[bestIdx])
}


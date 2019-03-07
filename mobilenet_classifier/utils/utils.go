package utils

//Necessary imports
import (
	"fmt"
	"os"
	"io/ioutil"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

//Paths to the serialized computational graph (pretrained MobileNet model) and list of labels for imagenet dataset
const ModelPath string = "/home/stefan/go/src/github.com/StefanCepa/MobileNetClassifier/mobilenet_classifier/mobilenet/model/mobilenet_v2_1.4_224_frozen.pb"
const LabelPath string = "/home/stefan/go/src/github.com/StefanCepa/MobileNetClassifier/mobilenet_classifier/mobilenet/labels/imagenet_labels.txt"

/**
Creates new execution session with associated graph
which is passed as an function argument

Arguments:
	- graph *tf.Graph: computational graph which is
			   suppose to be executed inside
		           the session

Return Value:
	- session *tf.Session: session object with associated
			       computational graph
**/
func CreateAndInitializeNewSession(graph *tf.Graph) *tf.Session{
	session, err := tf.NewSession(graph, nil)

	if err != nil {
		fmt.Printf("[ERROR]: Create and Initialize New Session\n" + err.Error())
		os.Exit(1)
	}

	return session
}

/**
Creates empty graph and then tries to import an array of
bytes which is our serialized representation of mobilenet
computational graph.

Arguments:
	- model []byte: array of bytes which represent our
	                pretrained model
Return Value:
	- graph *tf.Graph: if successfull, function returns a
		     computational graph which nodes and edges
		     were imported from function arguments
		     
**/
func CreateGraphAndImportExistingModel(model []byte) *tf.Graph{

	graph := tf.NewGraph()
	
	if err := graph.Import(model, ""); err != nil {
		fmt.Printf("[ERROR]: Importing Existing Model:\n%s",err.Error())
		os.Exit(1)
	}

	return graph
}


/**
This function utilizes another function, called ConstructGraphToNormalizeImage()
in order to create execution graph which is then used to initialize a session
in which it will be executed in order to normalize image passed as an argument

Argument:
	- filename string: URL of the image to be normalized

Return Value:
	- *tf.Tensor: image represented as tf.Tensor
	- err error: error if some exception occurs
**/
func ImageToTensor(filename string) (*tf.Tensor, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Printf("[ERROR]: Reading Image File\n" + err.Error())
		os.Exit(1)
	}
	
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		fmt.Printf("[ERROR]: Creating New Tensor Object\n" + err.Error())
		os.Exit(1)
	}
	
	graph, input, output, err := ConstructGraphToNormalizeImage()
	if err != nil {
		fmt.Printf("[ERROR]: Graph Construction for Nomralizing Image\n" + err.Error())
		os.Exit(1)
	}
	
	session := CreateAndInitializeNewSession(graph)

	defer session.Close()

	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)

	if err != nil {
		fmt.Printf("[ERROR]: Nomralizing Image\n" + err.Error())
		os.Exit(1)
	}

	return normalized[0], nil
}

/**
All operations in TensorFlow Go are executed within a session. First of
all, we need to construct computational graph from these operations before
we execute them within a session.

Arguments:
	- graph *tf.Graph: computational graph which is used for normalizing
			  images.
	- input: placeholder ofr image URL
	- output tf.Output: normalized image
	- err error: error message in case some exception occurs

Returns:
	- graph *tf.Graph: basically returns the address to the graph which
			  we passed as an argument, but the graph which is
			  stored on that memory address is not empty, it is 
			  consutructed and ready for execution
**/

func ConstructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 224, 224
	)
	
	s := op.NewScope() //Creates a new scope initialized with an empty graph
	input = op.Placeholder(s, tf.String)
	output = op.Div(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("scale"), float32(255)))

	graph, err = s.Finalize() //Finalize returns graph on which scope "s" operates on, or error if some exception occurs
	return graph, input, output, err
}


import * as React from 'react';
import { Text, View, StyleSheet, Image, Button, AppRegistry } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as tf from '@tensorflow/tfjs';

tf.setBackend('cpu');
export default class App extends React.Component {
  state = {
    image: null,
    predictions: null,
  };

  async componentDidMount() {
    // Load the model
    const model = await tf.loadLayersModel(`model/model.json`);
  }

  selectImage = async () => {
    try {
      const response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        base64: true,
      });
  
      if (!response.cancelled) {
        // Convert the image data to a numeric tensor
        const imageTensor = tf.tensor(response.base64, [224, 224, 3], 'float32');
  
        // Resize the image tensor
        const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
  
        // Set the image in the state and predict the class using the model shards
        this.setState({ image: response.uri, predictions: null }, async () => {
          const predictions = await this.predictWithShards(resized, model);
          this.setState({ predictions });
        });
      }
    } catch (error) {
      console.log(error);
    }
  };

  predictWithShards = async (input, model) => {
    // Load the weight shards
    const weights = [];
    for (let i = 1; i <= 11; i++) {
      const weight = await tf.loadWeights(`model/group1-shard${i}of11.bin`);
      weights.push(weight);
    }

    // Set the weights for each shard on the model
    model.setWeights(weights);

    // Make a prediction using the model
    const prediction = model.predict(input);

    // Return the prediction
    return prediction;
  }
  

  render() {

    const styles = StyleSheet.create({
      container: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
      },
      image: {
        width: 200,
        height: 200,
      },
    });
    const { image, predictions } = this.state;
    return (
      <View style={styles.container}>
      {image && (
        <React.Fragment>
          <Image source={{ uri: image }} style={styles.image} />
          <Text>Predicted class: {predictions && predictions[0]}</Text>
        </React.Fragment>
      )}
      <Button title="Select image" onPress={this.selectImage} />
    </View>    
    );
  }
}
AppRegistry.registerComponent('main', () => App);
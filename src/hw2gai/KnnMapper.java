package hw2gai;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KnnMapper extends Mapper<Object, Text, Text, IntWritable>{

	//  
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    
    KnnNode testNode;
    ArrayList<KnnNode> trainingData;
    
    
    public KnnMapper(Text word, KnnNode node, ArrayList<KnnNode> trainingData) {
		super();
		this.word = word;
		this.testNode = node;
		this.trainingData = trainingData;
	}
    
	public Text getWord() {
		return word;
	}

	public void setWord(Text word) {
		this.word = word;
	}

	public KnnNode getNode() {
		return testNode;
	}

	public void setNode(KnnNode node) {
		this.testNode = node;
	}

	public ArrayList<KnnNode> getTrainingData() {
		return trainingData;
	}

	public void setTrainingData(ArrayList<KnnNode> trainingData) {
		this.trainingData = trainingData;
	}

	public static IntWritable getOne() {
		return one;
	}

	public void mapDistance(){
		for(int i=0;i<trainingData.size();i++){
			trainingData.get(i).getNodeDistance(testNode);
		}
	}
	
	//setup the training data
	public void setUpTrainingData(){
		String inputFileTrainingDataPath = "iris_train.csv";
		trainingData = DataReader.parse(inputFileTrainingDataPath);
	}
	
	//
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
    
    	setUpTrainingData();
    	mapDistance();
    	
    	int k_default = 5;
    	
    	
    	HashSet<KnnNode> result = new HashSet<KnnNode>();
    	
    	int minIndex = 0;
    	
    	while(result.size() <= k_default){
    		KnnNode min = trainingData.get(0);
    		for(int i=0;i < trainingData.size();i++){
    			if(min.compareTo(trainingData.get(i))>0 ){
    				min = trainingData.get(i);
    				minIndex = i;
    			}
    		}
    		result.add(min);
    		trainingData.remove(minIndex);
    	}
    	
    	trainingData.addAll(result);
    	
    	Text newValue = new Text(result.toString());
    	
    	context.write(newValue, one);
    }
}

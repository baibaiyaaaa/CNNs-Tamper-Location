package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
//import java.lang.module.FindException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.Stack;

import project.Node;
import project.MerkleTree;
import util.HashGeneration;
/**
 * To be able to download a file I used 
 * https://stackoverflow.com/questions/20265740/how-to-download-a-pdf-from-a-given-url-in-java
 *
 */
public class Main {
public static void main(String[] args){
		
		
		MerkleTree m0 = new MerkleTree("data/bad.txt");		
		//String hash = m0.getRoot().getLeft().getRight().getData();
		//System.out.println(hash);
		
		m0.levelOrder(m0.getRoot());
		System.out.println();
		m0.InOrder_PrintLeaf(m0.getRoot());
		
		
		boolean valid = m0.checkAuthenticity("data/meta.txt");
		System.out.println();
		System.out.println(valid);
		if(valid == false){
			System.out.println();
			ArrayList<Stack<String>> corrupts = m0.findCorruptChunks("data/meta.txt");
			for(int i = 0 ; i < corrupts.size() ; i++) {
				System.out.println(corrupts.get(i).peek());
			}
			
			System.out.println();
			ArrayList<String> wrongHashes = new ArrayList<String>();
			for(int y = 0 ; y<corrupts.size() ; y++) {
				wrongHashes.add(corrupts.get(y).peek());
			}
			ArrayList<Node> wrongNodes = new ArrayList<Node>();
			wrongNodes = m0.findTheWrongFiles(wrongHashes);
			System.out.println(Arrays.toString(wrongNodes.toArray()));

			
			//System.out.println("Corrupt hash of first corrupt chunk is: " + corrupts.get(0).pop());
			//System.out.println("Corrupt hash of first corrupt chunk is: " + corrupts.get(0).pop());
		}
		
		
		// The following just is an example for you to see the usage. 
		// Although there is none in reality, assume that there are two corrupt chunks in this example.
		//ArrayList<Stack<String>> corrupts = m0.findCorruptChunks("data/1meta.txt");
		//System.out.println("Corrupt hash of first corrupt chunk is: " + corrupts.get(0).pop());
		//System.out.println("Corrupt hash of second corrupt chunk is: " + corrupts.get(1).pop());
		
		//download("secondaryPart/data/download_from_trusted.txt");
		
	}

	public static void download(String path){
		System.out.println("Processing...");
		Scanner input;
		try {
			input = new Scanner(new File(path));
			List<String> filesList = new ArrayList<String>();
			int count = 0;
			while(input.hasNextLine()) {
				String line = input.nextLine();
				if(!line.equals("")) {
					filesList.add(line);
					count++;
				}
			}
			File f = new File("secondaryPart/data/split");
			if(!f.exists()) {
				f.mkdir();
			}

			int round = count/3;
			for(int i = 1; i<=round ; i++) {
				File file = new File("secondaryPart/data/split/"+(i+3));
				if(!file.exists()) {
					file.mkdir();
				}
			}
			int i = 0;

			for(int j = 1;j<=round; j++) {
				Files.copy(new URL(filesList.get(i)).openStream(), Paths.get("secondaryPart/data/" + (j+3) + "meta.txt"), StandardCopyOption.REPLACE_EXISTING);
				i+=3;
			}

			i = 1;

			for(int j = 1;j<=round; j++) {
				Files.copy(new URL(filesList.get(i)).openStream(), Paths.get("secondaryPart/data/" +(j+3) + ".txt"), StandardCopyOption.REPLACE_EXISTING);
				i+=3;
			}


			i = 2;

			for(int j = 1;j<=round; j++) {
				Files.copy(new URL(filesList.get(i)).openStream(), Paths.get("secondaryPart/data/" + (j+3) + "alt.txt"), StandardCopyOption.REPLACE_EXISTING);
				i+=3;
			}	
			i=0;
			for(int a = 1 ; a<=round ; a++) {
				input = new Scanner(new File("secondaryPart/data/" + (a+3) + ".txt"));
				String temp = "";
				String lastTwo = "";
				while(input.hasNext()) {
					temp =input.nextLine();
					lastTwo = temp.substring(temp.length()-2);
					Files.copy(new URL(temp).openStream(), Paths.get("secondaryPart/data/split/"+(a+3)+"/"+lastTwo), StandardCopyOption.REPLACE_EXISTING);		
				}
				i++;
			}
			boolean isSame = false;
			for(int x = 1 ; x<= round ; x++) {
				PrintStream stream = new PrintStream(new File("secondaryPart/data/split/tempTextFile.txt"));
				File folder = new File("secondaryPart/data/split/"+ (x+3));
				File[] files = folder.listFiles();
				for(File file : files){
					stream.println(file.getPath());

				}
				MerkleTree newTree = new MerkleTree("secondaryPart/data/split/tempTextFile.txt");
				isSame = newTree.checkAuthenticity("secondaryPart/data/"+(x+3)+"meta.txt");
				if(isSame==false) {
					System.out.println("There are files in "+(x+3)+ ".txt that should be replaced with the correct ones!");
					System.out.println();
					ArrayList<Stack<String>> corruptOnes = newTree.findCorruptChunks("secondaryPart/data/"+(x+3)+"meta.txt");
					ArrayList<String> wrongHashes = new ArrayList<String>();
					for(int y = 0 ; y<corruptOnes.size() ; y++) {
						wrongHashes.add(corruptOnes.get(y).pop());
					}
					ArrayList<Node> wrongNodes = new ArrayList<Node>();
					wrongNodes = newTree.findTheWrongFiles(wrongHashes);
					if(!wrongNodes.isEmpty()) {
						newTree.changeWrongNodes(wrongNodes, x);
						System.out.println("Wrong files are changed.");
						System.out.println();
					}
				}
				else {
					System.out.println("All files in " + (x+3) + ".txt is correct.");
					System.out.println();
				}
			}


		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}

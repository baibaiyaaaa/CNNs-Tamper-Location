package filechunking;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class FileChunking {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		testChunk();
		testMerge();
		System.out.println("Program completed");
	}
	
	//�����ļ��ֿ鷽��
    //@Test
    public static void testChunk() throws IOException {
        File sourceFile = new File("data/vgg_16_bn.pt");
        String chunkPath = "data/split/";
        File chunkFolder = new File(chunkPath);
        if (!chunkFolder.exists()) {
            chunkFolder.mkdirs();
        }
        //�ֿ��С
        //long chunkSize = 1024 * 1024 * 1;
        long chunkSize = 1024 * 1024 * 5;
        //�ֿ�����
        long chunkNum = (long) Math.ceil(sourceFile.length() * 1.0 / chunkSize);
        if (chunkNum <= 0) {
            chunkNum = 1;
        }
        //��������С
        byte[] b = new byte[1024];
        //ʹ��RandomAccessFile�����ļ�
        RandomAccessFile raf_read = new RandomAccessFile(sourceFile, "r");
        //�ֿ�
        for (int i = 0; i < chunkNum; i++) {
            //�����ֿ��ļ�
            File file = new File(chunkPath + i);
            boolean newFile = file.createNewFile();
            if (newFile) {
                //��ֿ��ļ���д����
                RandomAccessFile raf_write = new RandomAccessFile(file, "rw");
                int len = -1;
                while ((len = raf_read.read(b)) != -1) {
                    raf_write.write(b, 0, len);
                    if (file.length() > chunkSize) {
                        break;
                    }
                }
                raf_write.close();
            }
        }
        raf_read.close();
    }
    
    
  //�����ļ��ϲ�����
    //@Test
    public static void testMerge() throws IOException {
        //���ļ�Ŀ¼
        File chunkFolder = new File("data/split/");
        //�ϲ��ļ�
        File mergeFile = new File("data/vgg_16_bn1.pt");
        if (mergeFile.exists()) {
            mergeFile.delete();
        }
        //�����µĺϲ��ļ�
        mergeFile.createNewFile();
        //����д�ļ�
        RandomAccessFile raf_write = new RandomAccessFile(mergeFile, "rw");
        //ָ��ָ���ļ�����
        raf_write.seek(0);
        //������
        byte[] b = new byte[1024];
        //�ֿ��б�
        File[] fileArray = chunkFolder.listFiles();
        //  ת�ɼ��ϣ���������
        List<File> fileList = new ArrayList<File>(Arrays.asList(fileArray));
        //  ��С��������
        Collections.sort(fileList, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                if (Integer.parseInt(o1.getName()) < Integer.parseInt(o2.getName())) {
                    return -1;
                }
                return 1;
            }
        });
        //�ϲ��ļ�
        for (File chunkFile : fileList) {
            RandomAccessFile raf_read = new RandomAccessFile(chunkFile, "rw");
            int len = -1;
            while ((len = raf_read.read(b)) != -1) {
                raf_write.write(b, 0, len);
            }
            raf_read.close();
        }
        raf_write.close();
    }



}

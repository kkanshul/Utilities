#include<stdio.h>
#include<utility>
#include<string>
#include<vector>
#include<algorithm>
#include<stdlib.h>
#include<chrono>
#include<iostream>

std::vector< std::vector<std::pair <int, float> > > features;
std::vector< std::vector <int> > test_nn;
std::vector < int > video_id;
//int NUM_NEIGHBOURS = 5;
int TRAINING_NUM = 0;
int TIME_GAP = 1200 * 5;
//int TIME_WINDOW = 10 * 5; //20 * 5;
int START_FRAME = 0;
int END_FRAME = 0;

typedef std::pair<int,float> mypair;
bool comparator ( const mypair& l, const mypair& r)
{ return l.second > r.second; }

// load deep features vectors and video clip IDs
void loadFeaturesAndIds(const char *featuresFilename, const char *videoIdFilename)
{
    // Read the binary file and store the features
    FILE *f = fopen(featuresFilename,"rb");
    int total_features;
    fread(&total_features,sizeof(int),1,f);
    printf("%d",total_features);
    //total_features = 10000;
    features.resize(total_features);
    for(int i = 0; i < total_features; i++)
    {
        int total_elements;
        fread(&total_elements,sizeof(int), 1, f);
        if(i%10000 == 0)
        {
//            printf("%d %d\n",i,total_elements);
        }
        std::vector< std::pair <int,float> > temp(total_elements);
        for (int j = 0; j < total_elements; j++)
        {
            int index; float val;
            fread(&index, sizeof(int), 1, f);
            fread(&val, sizeof(float), 1, f);
            temp[j].first = index - 1;
            temp[j].second = val;
            
        }
        features[i] = temp;
    }
    fclose(f);
    
    FILE *f1 = fopen(videoIdFilename, "rb");
    fread(&total_features,sizeof(int), 1, f1);
    printf("TOTAL_FEATURES = %d\n",total_features);
    video_id.resize(total_features);
    fread(&TRAINING_NUM,sizeof(int), 1, f1);
    printf("TRAINING_NUM = %d\n",TRAINING_NUM);
    //TRAINING_NUM = 5000;
    for(int i = 0; i < total_features; i++)
    {
        int curr_id;
        fread(&curr_id, sizeof(int), 1, f1);
        video_id[i] = curr_id;
    }
    fclose(f1);
}

// Cosine Similarity computation
float find_cosine(std::vector< std::pair <int,float> > &v1, std::vector< std::pair <int,float> > &v2)
{
    float dot = 0;
    unsigned int index_v1 = 0, index_v2 = 0;
    while (index_v1 < v1.size() && index_v2 < v2.size())
    {
        if (v1[index_v1].first == v2[index_v2].first)
            dot += v1[index_v1++].second * v2[index_v2++].second;
        else if (v1[index_v1].first < v2[index_v2].first)
            ++index_v1;
        else
            ++index_v2;
    }
    return dot;
}

// Nearest neighbors computation
void compute_nn(int query_ind, int NUM_NEIGHBORS, int TIME_WINDOW, std::vector <int> &NN, int randSize, int flag)
{
   
    int nbSize = (TIME_WINDOW + 1) * NUM_NEIGHBORS + randSize;
    std::vector < int > train_ind (nbSize);
    int curr_cnt_train = 0;
    int start_ind, end_ind;
    
    //start_ind  =  1;
    //end_ind  =  TIME_WINDOW / 2;
    if(flag == 0)
    {
        start_ind = -1 * TIME_WINDOW;
        end_ind = 0;
    }
    else
    {
        start_ind = 1;
        end_ind = TIME_WINDOW;
    }
    
    for (int i = start_ind; i < end_ind; i++)
    {
        int curr_ind = query_ind + i;
        //printf("%d %d\n",i,curr_ind);
        if(curr_ind >= START_FRAME && curr_ind < END_FRAME)
        {
            for(int j = 0; j < NUM_NEIGHBORS; j++)
            {
                int ind_added = test_nn[curr_ind-START_FRAME][j] - i;
                if(ind_added >= 0 && ind_added < TRAINING_NUM)
                {
                    train_ind[curr_cnt_train++] = ind_added;
                    //train_ind[curr_cnt_train++] = (rand() % TRAINING_NUM);
                }
            }
        }
    }
   
    // also include current frame's neighbours
    for (int j = 0; j < NUM_NEIGHBORS; j++)
    {
        train_ind[curr_cnt_train++] = test_nn[query_ind - START_FRAME][j];
    }
   
    // Randomly select randSize frames
    for (int i = 0; i < randSize; i++)
    {
        train_ind[curr_cnt_train++] = rand() % TRAINING_NUM;
    }
    
    //printf("done 1\n");
    train_ind.resize(curr_cnt_train);
    std::vector< std::pair <int,float> > cosine_similarity(curr_cnt_train);
    for(int i = 0; i < curr_cnt_train; i++)
    {
        cosine_similarity[i].first = train_ind[i];
        cosine_similarity[i].second = find_cosine(features[query_ind], features[train_ind[i]]);
        //printf("%d\n",i);
    }
    //printf("done %d\n", curr_cnt_train);
    std::sort(cosine_similarity.begin(), cosine_similarity.end(), comparator);
    int nn_cnt = 0, curr_cnt = 0;
    std::vector < int > nn_vid_id(NUM_NEIGHBORS);
    std::vector < int > nn_frame_id(NUM_NEIGHBORS);
    //printf("done-1 %d\n",curr_cnt_train);
    
    
    while(nn_cnt != NUM_NEIGHBORS && curr_cnt < curr_cnt_train)
    {
        int flag = 0;
        int curFrameId = cosine_similarity[curr_cnt].first;
        int curVideoId = video_id[curFrameId];
        for(int x = 0; x < nn_cnt; x++)
        {
            //printf("curFrameId = %d, nn_frameId[%d] = %d, difference = %d, ", curFrameId, x, nn_frame_id[x], abs(curFrameId - nn_frame_id[x]));
            if (nn_vid_id[x] == curVideoId && (abs(curFrameId - nn_frame_id[x]) < TIME_GAP))
            {
                flag = 1;
                //printf("flag = %d\n", flag);
                break;
            }
            //printf("flag = %d\n", flag);
        }
        if(flag == 0)
        {
            NN[nn_cnt] = curFrameId;
            nn_vid_id[nn_cnt] = curVideoId;
            nn_frame_id[nn_cnt] = curFrameId;
            nn_cnt++;
  //          printf("videoId = %d, frameId = %d, similarity = %f\n", curVideoId, curFrameId, cosine_similarity[curr_cnt].second);
        }
        curr_cnt++;
    }
   
    if (nn_cnt < NUM_NEIGHBORS)
    {
        printf("Odd case, we didn't find as many distinct nn as we wanted, found %d, wanted %d, randomly selecting frames \n", nn_cnt, NUM_NEIGHBORS);
        while (nn_cnt != NUM_NEIGHBORS)
        {
            NN[nn_cnt++] = rand() % TRAINING_NUM;
        }
    }
   // printf("done\n");
   // printf("\n");
}

void approxKNN_oneIter(int it, int randSize, int NUM_NEIGHBORS, int TIME_WINDOW, FILE *fout)
{
    int even_odd_flag = it % 2;
    // propagate backwards in even iterations and forwards in odd iterations
    int startIdx = END_FRAME - 1;
    int endIdx = START_FRAME;
    int incrementBy = -1;
    if (even_odd_flag)
    {
        startIdx = START_FRAME;
        endIdx = END_FRAME;
        incrementBy = 1;
    }
    int i = startIdx;
    for (int iter = 0; iter < END_FRAME - START_FRAME; iter++)
    {
     //   printf("Frame %d\n", i);
        std::vector <int> NN(NUM_NEIGHBORS);
        compute_nn(i, NUM_NEIGHBORS, TIME_WINDOW, NN, randSize, even_odd_flag);
        fprintf(fout, "%d", i);
        for(int j = 0; j < NUM_NEIGHBORS; j++)
        {
            fprintf(fout, " %d", NN[j]);
            test_nn[i-START_FRAME][j] = NN[j];
        }
        fprintf(fout,"\n");
        i += incrementBy;
    }
}

void approxNN(int iterations, int NUM_NEIGHBORS, int TIME_WINDOW, double randPercent, int test_size, const char *outputFilename, const char *outputFilename_stat)
{
    // Initialization Step:
    for (int i = 0; i < test_size;i++)
    {
        test_nn[i].resize(NUM_NEIGHBORS);
        for(int j = 0;j < NUM_NEIGHBORS;j++)
        {
            test_nn[i][j] =  (rand() % TRAINING_NUM);
        }
    }
    // Iteration Step:
    FILE *fout = fopen(outputFilename, "w");
    FILE *fout_stat = fopen(outputFilename_stat, "w");
    int randSize = int(randPercent * TIME_WINDOW * NUM_NEIGHBORS);
    printf("randSize = %d\n", randSize);
	//double total_time_elapse=0;
    for (int it = 0; it < iterations; it++)
    {
        printf("Algo iteration = %d\n", it);
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        approxKNN_oneIter(it, randSize, NUM_NEIGHBORS, TIME_WINDOW, fout);
        
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        //	total_time_elapse+=(double)elapsed_seconds.count();
        fprintf(fout_stat,"%d , %s , %f\n",it,std::ctime(&end_time),elapsed_seconds.count());
        std::cout << "Algo Iter = " << it << ", finished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }
    fclose(fout);
    fclose(fout_stat);
}

int main(int argc, char *argv[])
{
    
    // Read the binary file and store the features
    const char * featuresFilename = "../../data/binary_features.bin";
    const char * videoIdFilename = "../../data/binary_vidid.bin";
    //const char * featuresFilename = "/usr1/ksingh1/misc/binary_features.bin";
    //const char * videoIdFilename = "/usr1/ksingh1/misc/binary_vidid.bin";
    loadFeaturesAndIds(featuresFilename, videoIdFilename);
    
    // Compute nearest neighbors
    
    // Test frame range
    START_FRAME = atoi(argv[1]) + TRAINING_NUM;
    END_FRAME = atoi(argv[2]) + TRAINING_NUM;
    
    int test_size = END_FRAME - START_FRAME;
    test_nn.resize(test_size);
    //printf("s:%d e:%d\n", START_FRAME, END_FRAME);
    int wind_sec = atoi(argv[5]);
    const char *outputFilename = argv[3];
    const char *outputFilename_stat = argv[4];
    int NUM_NEIGHBORS = atoi(argv[6]);
    int TIME_WINDOW = 5 * wind_sec;// * 5;
    double randPercent = 0.75f;
   // int iterations = 50;//50;
    int iterations = atoi(argv[7]);//50;

    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    
    approxNN(iterations, NUM_NEIGHBORS, TIME_WINDOW, randPercent, test_size, outputFilename, outputFilename_stat);
   
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    
    std::cout << "finished computation at " << std::ctime(&end_time)
    << "elapsed time: " << elapsed_seconds.count() << "s\n";
    
    return 0;
}

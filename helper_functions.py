from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from scipy import interpolate
from keras.utils import plot_model

def dropout_search(dropout_ls):
    """
    Hyperparameter search for dropout
    """
    pass

def plot_model_and_save(self):
    pass

def plot_extractions_with_names(ext_images,ext_data,ext_class_name,ext_class_index,name,**kwargs):
    """ Should add here or in helper_functions ?? """
        #load kwarg ans, else leave ans variable as empty string
    ans = kwargs.get('ans','')

        #prepare figure size
    num_of_samples=len(ext_data)
    #include all input data in title
    subplot_num=int(np.ceil(np.sqrt(num_of_samples)))
    fig=plt.figure(figsize=(num_of_samples,num_of_samples))
    
    ##### Find Ground Truth Values if Available #####
    if ans != '':

        ans=np.array(ans)
        ext_class_index = np.array(ext_class_index)
        score_matrix = ans/ext_class_index
            # Replace all incorrect predictions with 0
        for i in range(0,len(ext_class_index)):
            if ans[i]==0 and ext_class_index[i]==0: #fix case of 0/0
                score_matrix[i]=1
            if score_matrix[i] != 1: #else change all non-1's (incorrect) to 0
                score_matrix[i] = 0
                
        li1= ['numbers','0','1','2','3','4','5','6','7','8','9']
        li2= ['forces','up','down','right','left']
        li3= ['moments','ctrcl_moments','cl_moments']
        li4= ['random','noise','alphab']
        li5= ['supports']
        li6= ['fixed_supports','fixed_right','fixed_left','fixed_down','fixed_up']
        li7= ['pinned_supports','pinned_down','pinned_up','pinned_left','pinned_right']
        li8= ['roller_supports','roller_down','roller_up','roller_left','roller_right']
        li9= ['distributed_loads', 'uniform_distributed','linear_distributed','quadratic_distributed','cubic_distributed']
        li10=['beams','horizontal','vertical','downward_diagonal','upward_diagonal']
        li11=['dimensions','length','height','ctrcl_angle','cl_angle']
            #create empty group lists 
        for j in (li1,li2,li3,li4,li5,li6,li7,li8,li9,li10,li11):
            for l in j:
                exec(str('acc_'+str(l))+'= []')

            #include scores into individual categories
        for i in range(1,len(ext_data)+1):
            if ans[i-1]>=0 and ans[i-1]<=9:
                exec('acc_'+'numbers'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li1)):
                    exec('if ans[i-1]=='+str(j-1)+': '+'acc_'+str(li1[j])+'.append(score_matrix[i-1])')
                        
            if ans[i-1]>=10 and ans[i-1]<=13:
                exec('acc_'+'forces'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li2)):
                    exec('if ans[i-1]=='+str(j+9)+': '+'acc_'+str(li2[j])+'.append(score_matrix[i-1])')
                    
            if ans[i-1]>=14 and ans[i-1]<=15:
                exec('acc_'+'moments'+'.append(score_matrix[i-1])')

                
                for j in range(1,len(li3)):
                    exec('if ans[i-1]=='+str(j+13)+': acc_'+str(li3[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=16 and ans[i-1]<=17:
                exec('acc_'+'random'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li4)):
                    exec('if ans[i-1]=='+str(j+15)+': acc_'+str(li4[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=18 and ans[i-1]<=21:
                exec('acc_'+'supports'+'.append(score_matrix[i-1])')
                
                if ans[i-1]==18:
                    exec('acc_'+'fixed_supports'+'.append(score_matrix[i-1])')
                if ans[i-1]==19:
                    exec('acc_'+'pinned_supports'+'.append(score_matrix[i-1])')
                if ans[i-1]==20 or ans[i-1]==21:
                    exec('acc_'+'roller_supports'+'.append(score_matrix[i-1])')
            if ans[i-1]>=22 and ans[i-1]<=25:
                exec('acc_'+'distributed_loads'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li9)):
                    exec('if ans[i-1]=='+str(j+21)+': acc_'+str(li9[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=26 and ans[i-1] <=29:
                exec('acc_'+'beams'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li10)):
                    exec('if ans[i-1]=='+str(j+25)+': acc_'+str(li10[j])+'.append(score_matrix[i-1])')
            
            if ans[i-1]>=30 and ans[i-1] <=33:
                exec('acc_'+'dimensions'+'.append(score_matrix[i-1])')
                
                for j in range(1,len(li11)):
                    exec('if ans[i-1]=='+str(j+29)+': acc_'+str(li11[j])+'.append(score_matrix[i-1])')
            
            #calculate accuracies by doing sum of 1's (correct answers) divided by number of entries in that category
        for j in (li1,li2,li3,li4,li5,li6,li7,li8,li9,li10,li11):
            for l in j:
                exec('try: acc1_'+str(l)+'= str(round((sum(eval(str(acc_'+str(l)+')))/len(eval(str(acc_'+str(l)+'))))*100,2))'+'\n'+'except:acc1_'+str(l)+"='N/A'")
            
            #prepare a list of "accuracy_x = some value" strings
        s=[]
        for j in (li1,li2,li3,li4,li5,li6,li7,li8):
            for l in j:
                s.append("acc_"+str(l)+" = "+str(eval("acc1_" +str(l)))+"% ")
            s.append("\n") #start new line after each category
            # join accuracy strings with spaces
        string=" ".join(s)

        ##### Calculate Overall Accuracy #####
        acc= sum(score_matrix)/len(score_matrix)
        print('Accuracy is '+str(acc*100)+" %")
        plt.title("Accuracy is "+str(acc*100)+" %" "\n" +string ,fontsize=20,color='blue')
    else:
        string='' #else, do anything useless
    
        #plot images with matching percentages, change to red if incorrect
    for i in range(1,len(ext_data)+1):
        color='black'
        if ans != '': #only do change to red if ans list is available
            if score_matrix[i-1] == 0:
                color='red'
            else:
                color='black'
        ax = fig.add_subplot(subplot_num, subplot_num, i)
        temp_image = ext_images[i-1]
            #plot extractions with matching percentages and corresponding color of text
        ax.set_title(str(i)+" prediction is : '"+ ext_class_name[i-1]+"' " "\n" + str(np.ceil(ext_match_percent[i - 1]*100*100)/100)+"% match" "\n" + str(((ext_match_percent[i-1]-ext_match_percent2[i-1])*100*100)/100) +"  1-2 difference %", color=color)
        ax.imshow(temp_image)
#        plt.tight_layout()
    
    plt.tight_layout()
    plt.show()
    
        #save the figure by name

    try:
        fig.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/ext'+str(name)+'.jpg')
        print('Saved extractions figure as /home/chloong/Desktop/Justin San Juan/Testing Folder/Output/ext'+str(name)+'.jpg')
    except:
        fig.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Output/ext'+str(name)+'.jpg')
        print('Saved extractions figure as C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Output/ext'+str(name)+'.jpg')

        # draw rectangles on the original image
    fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(25, 25))
    
    
    together = []
    for i in range(len(ext_data)):
        x, y, w, h = ext_data[i]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
        color='b' #not compared to any answer
        try:
            if ext_class_index[i] != ans[i]:
                color='r' # incorrect
            else:
                color = 'g' # correct
                
        except:
            color = 'b' #not compared to any answer
#        ax1.annotate(str(i) + ' : ' + str(ext_class_name[i]),xy=(x, y-2),fontsize=12,color=color)
        string = str(i+1) + ' : ' + str(ext_class_name[i])
        
        together.append((string,x,y,color))
    
    together.sort()

    text = [a for (a,b,c,d) in together]
    eucs = [b for (a,b,c,d) in together]
    covers = [c for (a,b,c,d) in together]
    cols = [d for (a,b,c,d) in together]

    plt.plot(eucs,covers,color="black", alpha=0.0)
    texts = []
    for xt, yt, s, color in zip(eucs, covers, text,cols):
        txt = plt.text(xt, yt, s,size = 18,color=color)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        texts.append(txt)

    f = interpolate.interp1d(eucs, covers)
    x = np.arange(min(eucs), max(eucs), 20)
    y = f(x)
    #y = f(x)
    #print(y)
    adjust_text(texts,x=x,y=y, arrowprops=dict(arrowstyle="->", color='r', lw=1),
                expand_points=(1.2,1.2),
                force_text=0.9,
                force_points = 0.15)
############ convert to graymap
    image1=np.multiply(np.subtract(image,1),-1)
    cmap = plt.cm.gray
    norm = plt.Normalize(image1.min(),image1.max())
    rgba = cmap(norm(image1))
    ax1.imshow(rgba,interpolation='nearest')
##################
#    ax1.imshow(image)
#    plt.tight_layout()
    plt.show()
    try:    
        fig1.savefig('C:/Users/JustinSanJuan/Desktop/HKUST/UROP Deep Learning Image-based Structural Analysis/Training Data Set/Output/print_'+str(name)+'.jpg')
    except: 
        fig1.savefig('/home/chloong/Desktop/Justin San Juan/Testing Folder/Output/print_'+str(name)+'.jpg')

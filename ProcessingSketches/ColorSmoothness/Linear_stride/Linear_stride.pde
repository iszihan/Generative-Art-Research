// Constants
int Y_AXIS = 1;
int X_AXIS = 2;
color b1, b2, c1, c2;
float nstep;

void setup() {
  size(318, 318);

  // Define colors
  

  noLoop();
}

void draw() {
  // Background
  //for(int itr = 1; itr <= 100; itr++){
   for(int level = 0; level < 10; level ++){
      //for(int rn = 0; rn <= 5; rn++){
    
        //int itr=1;
        //int level = width ; //2-width;
        int rn = 0;
        
        float r1 = 0;//random(0,100);
        float r2 = 255;//random(200,255);
        b1 = color(r1);
        b2 = color(r2);
        
        
        
        if(level == 0){
          nstep = width;
        }
        else if(level <= 3){
          nstep = map(level, 1, 3, 100, 60);
        }
        else if(level <= 6){
          nstep = map(level, 4, 6, 50, 20);
        }
        else if(level <= 9){
          nstep = map(level, 7, 9, 8, 2);
        }
        
        print("For level ", level, ",the nstep is: ",nstep, "\n");
        float smt = width/nstep;
        
        setGradient(0, 0, width, height, r2, r1, smt);
        
        PImage current = get();
        float rad = map(rn,0,5,0,TWO_PI);
        rotation(current, rad);
        PImage crop = get(47,47,224,224);
        String fileName = String.format("Output/tri-c%02d-%01d.png", level, rn);
        crop.save(fileName);
    }
   //}
  //}
}

void rotation(PImage img, float rad){
   pushMatrix();
   translate(width/2,height/2);
   rotate(rad);
   translate(-width/2,-height/2);
   image(img,0,0);
   popMatrix();
}


void setGradient(int x, int y, int w, int h, float b1, float b2, float smt) {

  
  c1 = color(b1);
  c2 = color(b2);
  
  //for (int i = x; i >= 0; i-=1) {
  //  //float inter = map(i, 0, x, 0, 1);
  //  //color c = lerpColor(b1, b2, inter);
  //  float temp_b = (x-i)/smt+b1;
  //  color c = color(temp_b);
  //  stroke(c);
  //  //fill(c);
  //  //rect(i-100,y,i,height);
  //  line(i, y, i, y+height);
  //}
  
  for (int i = x; i <= x+w; i+= smt) {
    float inter = map(i, x, x+w-smt, 0, 1);
    color c = lerpColor(c1, c2, inter);
    //float temp_b = (i-x)/smt+b1;
    //color c = color(temp_b);
    stroke(c);
    fill(c);
    rect(i,y,smt,h);
    line(i, y, i, y+h);
  }
  
  
}

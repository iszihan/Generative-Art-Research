// Constants
int Y_AXIS = 1;
int X_AXIS = 2;
color b1, b2, c1, c2;
float smt;


void setup() {
  size(500, 500);
  noLoop();
}

void draw() {
  
  //for(int itr = 1; itr <= 100; itr++){
  for(int level = 0; level < 10; level ++){
    //for(int rn = 0; rn <= 5; rn++){
        int itr=1;
        //int level = 1;
        int rn = 0;
        
        float r1 = 0;
        float r2 = 255;
        b1 = color(r1);
        b2 = color(r2);
        
        if(level <=6){
           smt = map(level,0,6,1,0.7);
        }
        else if(level <= 9){
           smt = map(level,7,9,0.6,0);
        }
        print("For level ", level, ",the smoothness is: ",smt, "\n");
        
        
        
        setGradient(width/2, 0, r1, r2, smt);
        
        PImage current = get();
        float rad = map(rn,0,5,0,TWO_PI);
        rotation(current, rad);
        PImage crop = get(138,138,224,224);
        String fileName = String.format("Output/tri-c%02d-%03d%01d.png", level, itr, rn);
        crop.save(fileName);
    }
  // }
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


void setGradient(int x, int y, float b1, float b2, float smt) {

  
  c1 = color(b1);
  c2 = color(b2);
  
  //unit distance is the same, adjusting color change per unit pixel distance
  for (int i = x; i >= 0; i-=1) {
    float temp_b = (x-i)+b1;
    float noise = random(0,255);
  
    color c = color(smt*temp_b+(1-smt)*noise);
    stroke(c);
    line(i, y, i, y+height);
  }
  
  for (int i = x; i <= width; i+=1) {
    float temp_b = (i-x)+b1;
    float noise = random(0,255);
    color c = color(smt*temp_b+(1-smt)*noise);
    stroke(c);
    line(i, y, i, y+height);
  }
  
  
}

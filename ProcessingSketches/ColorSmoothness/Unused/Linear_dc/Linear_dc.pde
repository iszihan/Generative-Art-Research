// Constants
int Y_AXIS = 1;
int X_AXIS = 2;
color b1, b2, c1, c2;


void setup() {
  size(500, 500);
  noLoop();
}

void draw() {
  
  //for(int itr = 1; itr <= 100; itr++){
   for(int level = 0; level < 10; level ++){
  //    for(int rn = 0; rn <= 5; rn++){
        int itr=1;
        //int level = 1;
        int rn = 0;
        
        float r1 = 0;
        float r2 = 255;
        b1 = color(r1);
        b2 = color(r2);
        
        float smt = map(level,0,9,1.5,10);
        //float smt=10;
        setGradient(width/2, 0, r1, r2, smt);
        //if (level <= 8){
        //  r1 = random(0,20);
        //  r2 = random(235,255);
         
          
        //  setGradient(width/2, 0, r1, r2, smt);
        //  setGradient(width/2, 0, r2, r1, smt);
        //}
        //else{
        //  //setGradient(width/2, 0, r1, r2, smt);
        //  setGradient(width/2, 0, r2, r1, smt);
        
        //}
        
        PImage current = get();
        float rad = map(rn,0,5,0,TWO_PI);
        rotation(current, rad);
        PImage crop = get(138,138,224,224);
        String fileName = String.format("Output/tri-c%02d-%03d%01d.png", level, itr, rn);
        crop.save(fileName);
  //  }
  }
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
    float temp_b = (x-i)*smt+b1;
    color c = color(temp_b);
    stroke(c);
    line(i, y, i, y+height);
  }
  
  for (int i = x; i <= width; i+=1) {
    float temp_b = (i-x)*smt+b1;
    color c = color(temp_b);
    stroke(c);
    line(i, y, i, y+height);
  }
  
  
}

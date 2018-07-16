int count = 0;
final float mess = 4;
final int iterations = 125;

void setup() {
  size(224, 224);
  noLoop();
}

void draw() {
  clear();
  float x;
  float y;
  float otherX;
  float otherY;
  float rotation;
  float scale;
  int choice;
  float background_col;
  
  for (int iteration = 0; iteration < iterations; iteration++) {  // Each set of 100 images
    for (int m = 1; m <= 10; m++) {  // Each of the images
      for(int blur = 0; blur <=3; blur++){
        
        //print(count + "\n");
        background_col = random(70,240);
        background(background_col);
        
        //if (count >= 2000) {
        //  print("Stopping at " + count + " images");
        //  stop();
        //}
        //count++;
        int level = m*m;
        if(m==1){
          level = 2;
        }
        
        scale = 10 + 40 * (level/100);
        for(int i = 0; i < (level-1) * mess + 2; i++) {  // Each primitive drawm
          
          x = random(15, width-15);
          y = random(15, height-15);
          otherX = random(x - 50, x + 50);
          while (otherX > x-10 && otherX < x+10){
            otherX = random(x - 50, x + 50);
          }
          otherY = random(y - 50, y + 50);
          while (otherY > y-10 && otherY < y+10){
            otherY = random(y - 50, y + 50);
          }
          rotation = random(0, TWO_PI);
          
          if (level > 20 && random(0,1) < 0.2) {
            noFill();
          }
          else {
            float col = random(50,255);
            while(col > background_col-15 && col < background_col+15 ){
              col = random(50,255);
            }
            fill(col);
          }
          stroke(random(0,150));
        
          pushMatrix();
          rotate((m*m/100) * random(-PI, PI));
          strokeWeight(1);
          choice = (int)random(1,6);  // Choice of which 2D primitive to draw
          //print("\nChoice: " + choice);
          //print("Made it to primitive choice, image #" + count);
          switch(choice) {
            case 1:  // Arc
              float start_rot = random(0,TWO_PI);
              while(start_rot == rotation){
                start_rot = random(0,TWO_PI);
              }
              arc(x, y, otherX, otherY, start_rot, rotation);
              break;
            case 2:  // Ellipse
              ellipse(x, y, random(5, scale), random(5, scale));
              break;
            case 3:  // Line
              line(x, y, otherX, otherY);
              break;
            case 4:  // Quad
              quad(x, y, otherX, otherY, random(x-scale, x+scale), random(y-scale, y+scale), random(x-scale, x+scale), random(y-scale, y+scale));
              break;
            case 5:  // Rect
              rect(x, y, random(15, scale*1.5), random(15, scale*1.5), random(0, 15));
              break;
            case 6: // Triangle
              triangle(x, y, otherX, otherY, random(x-30, x+30), random(y-30, y+30));
              break;
            default:
              print("Uh oh");
          }
          popMatrix();
          strokeWeight(2);
          stroke(random(0,50), 20);
          for (int point = 0; point < level; point++) {
            point(random(0,width), random(0,height));
          }
        }
      
      //print("Saving image with complexity %02d and at iteration %03d",m,iteration);
      PImage temp = get();
      temp.filter(BLUR,blur);
      temp.save(String.format("Output/tri-m%02d-mes%03d%01d.png", m, iteration, blur));
      }
    }
    //print(count);
  }
}

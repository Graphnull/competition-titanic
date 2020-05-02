let csvParse = require('csv-parse/lib/sync')
let fs = require('fs')
let train= csvParse(fs.readFileSync('./train.csv'))
//let test_raw= csvParse(fs.readFileSync('./test.csv'))

train = train.map(v=>{
    //v[3]=v[3].slice(0,1)
    v[9]=(parseFloat(v[9])/10)|0
    return v;
})
let test = train.slice(train.length-100)
test.unshift(train[0])
train=train.slice(0,train.length-100);

class PointsArray{
    constructor(points, inputDim,outputDim){
        this.constructorValidation(points, inputDim,outputDim);
        this.points=points;
        this.dim=inputDim+outputDim;
        this.inputDim=inputDim;
        this.outputDim=outputDim;
        this.count = points.length/this.dim
    }
    constructorValidation(points, inputDim,outputDim){
        if(typeof inputDim!=='number'||inputDim<1){
            throw new Error('typeof inputDim!=="number"||inputDim<1')
        }
        if(typeof outputDim!=='number'||outputDim<1){
            throw new Error('typeof outputDim!=="number"||outputDim<1')
        }
        let dim = inputDim+outputDim
        if(points.length%dim!==0){
            throw new Error('points.length%dim!==0')
        }
        
        if(!(points instanceof Float32Array) ){
            throw new Error('!(points instanceof Float32Array)')
        }
    }
    center(){
        if(!this._center){

            let sum = new Float32Array(this.dim)
            for(let i=0;i!==this.count;i++){
                for(let j=0;j!==this.dim;j++){
                    sum[j]+=this.points[i*this.dim+j]
                }
            }
            for(let j=0;j!==this.dim;j++){
                sum[j]/=this.count
            }
            this._center=sum
        }
        return this._center;
    }
    outputCenter(){
        let sum = new Float32Array(this.outputDim)
        for(let i=0;i!==this.count;i++){
            for(let j=0;j!==this.outputDim;j++){
                sum[j]+=this.points[i*this.dim+j+this.inputDim]
            }
        }
        for(let j=0;j!==this.outputDim;j++){
            sum[j]/=this.count
        }
        //console.log(sum);
        return sum;
    }
    angle(){
        let center = this.center()
        let distances = new Float32Array(this.dim)
        let offsets = new Float32Array(this.dim)
        for(let i=0;i!==this.count;i++){

            for(let j=0;j<this.dim;j++){
                distances[j]+=Math.pow(center[j]-this.points[i*this.dim+j],2)
            }
        }
        let maxDim=0;
        let maxDist =0;
        for(let j=0;j<this.dim;j++){
            if(distances[j]>maxDist){
                maxDist=distances[j];
                maxDim=j;
            }
        }
        for(let i=0;i!==this.count;i++){
            if(this.points[i*this.dim+maxDim]>center[maxDim]){
                for(let j=0;j<this.dim;j++){
                    offsets[j]+=this.points[i*this.dim+j]-center[j]
                }
            }
        }
        for(let i=0;i<this.dim;i++){
            if(offsets[i]<0){
                distances[i]=-distances[i]
            }
            distances[i]=distances[i]/maxDist;
        }
        return distances;
    }
    divineByAngle(){
        let angle = this.angle()
        let center = this.center();
        let leftCount = 0
        let rightCount = 0
        let results = new Uint8Array(this.count)
        for(let i=0;i!==this.count;i++){
            let sum =0;
            for(let j=0;j<this.dim;j++){
                sum+=(this.points[i*this.dim+j]-center[j])*angle[j]
            }
            if(sum>0){
                results[i]=1
                rightCount++
            }else{
                results[i]=0;
                leftCount++
            }
        }
        let left = new Float32Array(leftCount*this.dim)
        let right = new Float32Array(rightCount*this.dim)
        let leftOffset = 0;
        let rightOffset = 0;
        for(let i=0;i!==this.count;i++){
            if(results[i]){
                for(let j=0;j<this.dim;j++){
                    right[rightOffset*this.dim+j]=this.points[i*this.dim+j];
                }
                rightOffset++
            }else{
                for(let j=0;j<this.dim;j++){
                    left[leftOffset*this.dim+j]=this.points[i*this.dim+j];
                }
                leftOffset++
            }
        }
        return [new PointsArray(left,this.inputDim, this.outputDim),new PointsArray(right,this.inputDim, this.outputDim)]
    }
}


let convertXFunc = v=>{
    let pclass = [-1,-1,-1];
    pclass[v[2]-1]=1
    let Embarked =[-1,-1,-1,-1]
    if(v[11]==='C'){
     Embarked[0]=1
    }else if(v[11]==='Q'){
     Embarked[1]=1
    }else if(v[11]==='S'){
     Embarked[2]=1
    }else{
     Embarked[3]=1
    }
 let out =  pclass.concat([v[4]==='male'?1:-1,v[5]/70,v[9]/20])
 out=out.concat(Embarked)
 out.push(parseInt(v[6]))
 out.push(parseInt(v[7])/3)
 out.push(parseInt(v[1]))
 return out;
 //return [v[2]*0.3,v[4]==='male'?1:-1,v[5]/70,v[9]/20]
}

let p = new PointsArray(new Float32Array([].concat(...train.slice(1).map(convertXFunc))), 12,1)

let tree={
    points:p,
    output:p.outputCenter(),
    center:p.center(),
    angle:p.angle()
}
let div=(t)=>{
    //console.log(Object.keys(tree));
    getHit(tree)
    let arr = t.points.divineByAngle();
    if(arr[0].count>0&&arr[0].count!==t.points.count&&arr[1].count>0&&arr[1].count!==t.points.count){
        
        t[0]={
            points:arr[0],
            output:arr[0].outputCenter(),
            center:arr[0].center(),
            angle:arr[0].angle(),
        };
        div(t[0])
        t[1]={
            points:arr[1],
            output:arr[1].outputCenter(),
            center:arr[1].center(),
            angle:arr[1].angle(),
        };
        div(t[1])
    }else{
        //if(t.points.count>8){
        //console.log(t.points.points,t.points.count, arr[0].count, arr[1].count);
        //}
    }
}
let getOutput = (t, input)=>{
    if(t[0]&&t[1]){
        let sum =0;
        let angle = t.angle
        let center = t.center
        let dim = t.points.inputDim;
        for(let j=0;j<dim;j++){
            sum+=(input[j]-center[j])*angle[j]
        }
        if(sum>0){
            return getOutput(t[1],input)
        }else{
            return getOutput(t[0],input)
        }
    }else{
        return t.output
    }
}
let getViz = (t, obj)=>{
    if(t[0]){
        obj[0]={}
        obj[1]={}
        getViz(obj[0])
        getViz(obj[1])
        return obj;
    }else{
        return obj
    }
}
let getHit = (t)=>{
    let hit =0;
    let count = test.slice(1).length;
    
    test.slice(1).map(convertXFunc).forEach((value,i)=>{
        
        let out = getOutput(t, value)[0]
        if(value[12]>0.5){
            if(out>0.5) {hit++}
        }else{
            if(!(out>0.5)) {hit++}
        }

    })
    if(hit>70){
    console.log( (hit/count)*100,'%');
    }
}
div(tree);
console.log('sddd');
//console.log(getViz(tree));
getHit(tree)




let test_raw= csvParse(fs.readFileSync('./test.csv'))

    
    test_raw[0].unshift('PassengerId')
    test_raw[0][1]='Survived'
    let testout= [test_raw[0]].concat(test_raw.slice(1).map(v=>{
        v.unshift(v[0])
        return v;
    }))
    let validationDataX = testout.slice(1).map(convertXFunc)
    

let csvsave = require('csv/lib/index')
let result = [['PassengerId','Survived']]
validationDataX.forEach((value,i)=>{
    let out = getOutput(tree, value)[0]
    let surv =out>0.5?1:0
    result.push([testout.slice(1)[i][0],''+surv])
})
csvsave.stringify(result,(err, data)=>{
    fs.writeFileSync('./resulttree.csv', data)
})
//------------------------1. PREPARATION------------------------//
//-----------------------------SVG------------------------------//
const margin = 250
const global_height = 500 
const global_width = 960
const width = global_width - margin
const height = global_height - margin
// we are appending SVG first
const svg_a = d3.select("body")
        .append("svg")
        .attr("id","svg-a")
        .attr("width",global_width)
        .attr("height", global_height)

        
const g_a = svg_a.append("g")
        .attr("id","plot-a");


//Create second plot svg_b
const svg_b = d3.select("body")
            .append("svg")
            .attr("id","svg-b")
            .attr("width",global_width)
            .attr("height", global_height)

const g_b = svg_b.append("g")
        .attr("id","plot-b");

//Create third plot svg-c-1
const svg_c_1 = d3.select("body")
.append("svg")
.attr("id","svg-c-1")
.attr("width",global_width)
.attr("height", global_height)

const g_c_1 = svg_c_1.append("g")
.attr("id","plot-c-1");

//Create fourth plot svg-c-2
const svg_c_2 = d3.select("body")
.append("svg")
.attr("id","svg-c-2")
.attr("width",global_width)
.attr("height", global_height)

const g_c_2 = svg_c_2.append("g")
.attr("id","plot-c-2");


//-----------------------------DATA-----------------------------//
const timeConv = d3.timeParse("%Y-%m-%d");
const dataset = d3.dsv(",","boardgame_ratings.csv");
dataset.then(function(data) {
    var slices = data.columns.slice(1).map(function(id) {
        return {
            id: id,
            values: data.map(function(d){
                return {
                    date: timeConv(d.date),
                    measurement: +d[id]
                };
            })
        };
    });
    var countSlices = slices.filter(function (d) { return d.id.match(/count/); });
    var rankSlices = slices.filter(function (d) { return d.id.match(/rank/); })
    var rankPoints = slices.filter(function (d) { return d.id.match(/Catan=count|Codenames=count|Terraforming Mars=count|Gloomhaven=count/); })

//----------------------------SCALES----------------------------//
const xScale = d3.scaleTime().range([0,width]);
const yScale = d3.scaleLinear().rangeRound([height, 0]);
const yScaleSqrt = d3.scaleSqrt().rangeRound([height, 0]); 
const yScaleLog = d3.scaleLog().rangeRound([height, 1]); 
const colors = d3.scaleOrdinal(d3.schemeCategory10);

xScale.domain(d3.extent(data, function(d){
    return timeConv(d.date)}));

yScale.domain([(0), d3.max(countSlices, function(c) {
    return d3.max(c.values, function(d) {
        return d.measurement + 4; });
        })
    ]);

    

yScaleSqrt.domain([(0), d3.max(countSlices, function(c) {
    return d3.max(c.values, function(d) {
        return d.measurement; });
        })
    ])

yScaleLog.domain([(0), d3.max(countSlices, function(c) {
    return d3.max(c.values, function(d) {
        return d.measurement; });
        })
    ])

    

//-----------------------------AXES-----------------------------//
const yaxis = d3.axisLeft()
    //.ticks((slices[0].values).length)
    .scale(yScale);

const yaxisSqrt = d3.axisLeft()
    //.ticks((slices[0].values).length)
    .scale(yScaleSqrt);    


const yaxisLog = d3.axisLeft()
    //.ticks((slices[0].values).length)
    .scale(yScaleLog);    

const xaxis = d3.axisBottom()
    .ticks(d3.timeMonth.every(3))
    .tickFormat(d3.timeFormat('%b %y'))
    .scale(xScale);

//----------------------------LINES-----------------------------//
const line = d3.line()
    .x(function(d) { return xScale(d.date)+100; })
    .y(function(d) { return yScale(d.measurement)+50; });

const lineSqrt = d3.line()
    .x(function(d) { return xScale(d.date)+100; })
    .y(function(d) { return yScaleSqrt(d.measurement)+100; });    

const lineLog = d3.line()
    .x(function(d) { return xScale(d.date)+100; })
    .y(function(d) { return yScaleLog(d.measurement)+100; });    


let id = 0;
const ids = function () {
    return "line-"+id++;
}  

const point_ids = function () {
    return "point-"+id++;
}  


//---------------------------TOOLTIP----------------------------//


//-------------------------2. DRAWING---------------------------//
//-----------------------------AXES-----------------------------//

g_a.append("text")
    .attr("id","title-a")
    .attr("transform", "translate(300,0)")
    .attr("y", 25)
    .attr("font-size", "20px")
    .text("Number of Ratings 2016-2020");
    
g_a.append("g")
    .attr("transform", "translate(100," + (height+50) + ")")
    .attr("id", "x-axis-a")
    .call(xaxis);

g_a.select("#x-axis-a")
    .append("text")
    .attr("id", "x-axis label")
    .attr("y", height - 200)
    .attr("x", width - 350)
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Month");

g_a.append("g")
    .attr("id", "y-axis-a")
    .attr("transform", "translate(100,50)")
    .call(yaxis);

g_a.select("#y-axis-a")
    .append("text")
    .attr("id", "y-axis label")
    .attr("transform", "rotate(-90)")
    .attr("y", height - 250)
    .attr("x", -100)
    .attr("dy", "-5.1em")
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Num of Ratings");

//----------------------------LINES-----------------------------//

const countLinesA = svg_a.select("#plot-a").append("g").attr("id","lines-a")
    .selectAll("lines")
    .data(countSlices)
    .enter();

countLinesA.append("path")
    .attr("class", ids)
    .attr("fill", "none")
    .attr("d", function(d) { return line(d.values); })
    .style("stroke", function(d, i) {
    return colors(d.id);
});

countLinesA.append("text")
    .attr("class","serie_label")
    .datum(function(d) {
        return {
            id: d.id,
            value: d.values[d.values.length - 1]}; })          
    .attr("transform", function(d) {
            return "translate(" + (xScale(d.value.date) + 100)  
            + "," + (yScale(d.value.measurement) + 55 )+ ")"; })
    .attr("x", 5)
    .text(function(d) { return d.id.split('=')[0]; })
    .style("fill", function(d, i) {
    return colors(d.id);});

//------------------------ PLOT-B------------------------//
//-------------------------1. DRAWING---------------------------//
//-----------------------------AXES-----------------------------//

// Draw the axis and labels for plot-b
g_b.append("text")
    .attr("id","title-b")
    .attr("transform", "translate(300,0)")
    .attr("y", 25)
    .attr("font-size", "20px")
    .text("Number of Ratings 2016-2020 with Rankings");
    
g_b.append("g")
    .attr("transform", "translate(100," + (height+50) + ")")
    .attr("id", "x-axis-b")
    .call(xaxis);

g_b.select("#x-axis-b")
    .append("text")
    .attr("id", "x-axis label")
    .attr("y", height - 200)
    .attr("x", width - 350)
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Month");

g_b.append("g")
    .attr("id", "y-axis-b")
    .attr("transform", "translate(100,50)")
    .call(yaxis);

g_b.select("#y-axis-b")
    .append("text")
    .attr("id", "y-axis label")
    .attr("transform", "rotate(-90)")
    .attr("y", height - 250)
    .attr("x", -100)
    .attr("dy", "-5.1em")
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Num of Ratings");

//----------------------------LINES-----------------------------//

const countLinesB = svg_b.select("#plot-b").append("g").attr("id","lines-b")
    .selectAll("lines")
    .data(countSlices)
    .enter();

countLinesB.append("path")
    .attr("class", ids)
    .attr("fill", "none")
    .attr("d", function(d) { 
        return line(d.values); 
    })
    .style("stroke", function(d, i) {
    return colors(d.id);
});

countLinesB.append("text")
    .attr("class","serie_label")
    .datum(function(d) {
        return {
            id: d.id,
            value: d.values[d.values.length - 1]}; })          
    .attr("transform", function(d) {
            return "translate(" + (xScale(d.value.date) + 100)  
            + "," + (yScale(d.value.measurement) + 55 )+ ")"; })
    .attr("x", 5)
    .text(function(d) { return d.id.split('=')[0]; })
    .style("fill", function(d, i) {
    return colors(d.id);});

rankSlices.forEach(function(d,i) {
    rankSlices[i]['values']=
    d.values.filter(function(d,i){
            return (i+1)%3==0
        });
    });

rankPoints.forEach(function(d,i) {
    rankPoints[i]['values']=
    d.values.filter(function(d,i){
        return (i+1)%3==0
        });
    });

for (let i = 0; i < rankPoints.length; i++) {
    values_arr = rankPoints[i].values
    for (let j = 0; j < values_arr.length; j++) {
        values_arr[j]["rank"] = rankSlices[i].values[j].measurement
        values_arr[j]["id"] = rankPoints[i].id
    }
}

var mapValues = []
for (let i = 0; i < rankPoints.length; i++) {
    values_arr = rankPoints[i].values
    mapValues.push(values_arr)
}

svg_b.select("#plot-b")
    .append("g")
    .attr("id","symbols-b")

var  pointsLinesB = svg_b.select("#symbols-b")
    .selectAll("g")
    .data(rankPoints)
    .enter()
    .append("g")

var circles = pointsLinesB.selectAll(".circles")
    .data(function(d){
        return d.values
    })
    .enter()
    .append("circle")
    
circles.attr("cx", function(d) { 
        return xScale(d.date)+110; })      
    .attr("cy", function(d,i) {
        return yScale(d.measurement)+45; })    
    .attr("r", 14)
    .style("fill", function(d) {
        return colors(d.id);
    }) 

var circles_text = pointsLinesB.selectAll(".text")
    .data(function(d){
        return d.values
    })
    .enter()
    .append("text")

circles_text.attr("x", function(d) { 
        return xScale(d.date)+100; })      
    .attr("y", function(d,i) {
        return yScale(d.measurement)+50; })     
    .text(function(d) {
        return d.rank;})
    .style('fill', 'white');

legend = svg_b.append("g")
    .attr("id","legend-b")

legend.append("circle").attr("cx",width+200).attr("cy",300).attr("r", 20).style("fill", "black")
legend.append("text").attr("x", width+190).attr("y", 300).text("rank").style("font-size", "12px").style("fill","white")
legend.append("text").attr("x", width+150).attr("y", 330).text("BoardGameGeek Rank").style("font-size", "13px").style("fill","black")


//------------------------ PLOT-C1------------------------//
//-------------------------1. DRAWING---------------------------//
//-----------------------------AXES-----------------------------//

// Draw the axis and labels for plot-b
g_c_1.append("text")
    .attr("id","title-c-1")
    .attr("transform", "translate(300,0)")
    .attr("y", 25)
    .attr("font-size", "20px")
    .text("Number of Ratings 2016-2020 (Square Root Scale)");
    
g_c_1.append("g")
    .attr("transform", "translate(100," + (height+50) + ")")
    .attr("id", "x-axis-c-1")
    .call(xaxis);

g_c_1.select("#x-axis-c-1")
    .append("text")
    .attr("id", "x-axis label")
    .attr("y", height - 200)
    .attr("x", width - 350)
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Month");

g_c_1.append("g")
    .attr("id", "y-axis-c-1")
    .attr("transform", "translate(100,50)")
    .call(yaxisSqrt);

g_c_1.select("#y-axis-c-1")
    .append("text")
    .attr("id", "y-axis label")
    .attr("transform", "rotate(-90)")
    .attr("y", height - 250)
    .attr("x", -100)
    .attr("dy", "-5.1em")
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Num of Ratings");

//----------------------------LINES-----------------------------//

const countLinesC1 = svg_c_1.select("#plot-c-1").append("g").attr("id","lines-c-1")
    .selectAll("lines")
    .data(countSlices)
    .enter();

countLinesC1.append("path")
    .attr("class", ids)
    .attr("fill", "none")
    .attr("d", function(d) { 
        return lineSqrt(d.values); 
    })
    .style("stroke", function(d, i) {
    return colors(d.id);
});

countLinesC1.append("text")
    .attr("class","serie_label")
    .datum(function(d) {
        return {
            id: d.id,
            value: d.values[d.values.length - 1]}; })          
    .attr("transform", function(d) {
            return "translate(" + (xScale(d.value.date) + 100)  
            + "," + (yScale(d.value.measurement) + 55 )+ ")"; })
    .attr("x", 5)
    .text(function(d) { return d.id.split('=')[0]; })
    .style("fill", function(d, i) {
    return colors(d.id);});

svg_c_1.select("#plot-c-1")
    .append("g")
    .attr("id","symbols-c-1")

var  pointsLinesC1 = svg_c_1.select("#symbols-c-1")
    .selectAll("g")
    .data(rankPoints)
    .enter()
    .append("g")

var circlesC1 = pointsLinesC1.selectAll(".circles")
    .data(function(d){
        return d.values
    })
    .enter()
    .append("circle")
    
circlesC1.attr("cx", function(d) { 
        return xScale(d.date)+110; })      
    .attr("cy", function(d,i) {
        return yScale(d.measurement)+45; })    
    .attr("r", 14)
    .style("fill", function(d) {
        return colors(d.id);
    }) 

var circles_text_c1 = pointsLinesC1.selectAll(".text")
    .data(function(d){
        return d.values
    })
    .enter()
    .append("text")

circles_text_c1.attr("x", function(d) { 
        return xScale(d.date)+100; })      
    .attr("y", function(d,i) {
        return yScaleSqrt(d.measurement)+50; })     
    .text(function(d) {
        return d.rank;})
    .style('fill', 'white');


legendc1 = svg_c_1.append("g")
.attr("id","legend-c-1")

legendc1.append("circle").attr("cx",width+200).attr("cy",300).attr("r", 20).style("fill", "black")
legendc1.append("text").attr("x", width+190).attr("y", 300).text("rank").style("font-size", "12px").style("fill","white")
legendc1.append("text").attr("x", width+150).attr("y", 330).text("BoardGameGeek Rank").style("font-size", "13px").style("fill","black")        
    
//------------------------ PLOT-C2------------------------//
//-------------------------1. DRAWING---------------------------//
//-----------------------------AXES-----------------------------//

// Draw the axis and labels for plot-b
g_c_2.append("text")
    .attr("id","title-c-2")
    .attr("transform", "translate(300,0)")
    .attr("y", 25)
    .attr("font-size", "20px")
    .text("Number of Ratings 2016-2020 (Square Root Scale)");
    
g_c_2.append("g")
    .attr("transform", "translate(100," + (height+50) + ")")
    .attr("id", "x-axis-c-2")
    .call(xaxis);

g_c_2.select("#x-axis-c-2")
    .append("text")
    .attr("id", "x-axis label")
    .attr("y", height - 200)
    .attr("x", width - 350)
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Month");

g_c_2.append("g")
    .attr("id", "y-axis-c-2")
    .attr("transform", "translate(100,50)")
    .call(yaxisLog);

g_c_2.select("#y-axis-c-2")
    .append("text")
    .attr("id", "y-axis label")
    .attr("transform", "rotate(-90)")
    .attr("y", height - 250)
    .attr("x", -100)
    .attr("dy", "-5.1em")
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Num of Ratings");

//----------------------------LINES-----------------------------//

const countLinesC2 = svg_c_2.select("#plot-c-2").append("g").attr("id","lines-c-2")
    .selectAll("lines")
    .data(countSlices)
    .enter();

countLinesC2.append("path")
    .attr("class", ids)
    .attr("fill", "none")
    .attr("d", function(d) { 
        return lineLog(d.values); 
    })
    .style("stroke", function(d, i) {
    return colors(d.id);
});

countLinesC2.append("text")
    .attr("class","serie_label")
    .datum(function(d) {
        return {
            id: d.id,
            value: d.values[d.values.length - 1]}; })          
    .attr("transform", function(d) {
            return "translate(" + (xScale(d.value.date) + 100)  
            + "," + (yScale(d.value.measurement) + 55 )+ ")"; })
    .attr("x", 5)
    .text(function(d) { return d.id.split('=')[0]; })
    .style("fill", function(d, i) {
    return colors(d.id);});

svg_c_2.select("#plot-c-2")
    .append("g")
    .attr("id","symbols-c-2")

var  pointsLinesC2 = svg_c_2.select("#symbols-c-2")
    .selectAll("g")
    .data(rankPoints)
    .enter()
    .append("g")

var circlesC2 = pointsLinesC2.selectAll(".circles")
    .data(function(d){
        return d.values
    })
    .enter()
    .append("circle")
    
circlesC2.attr("cx", function(d) { 
        return xScale(d.date)+110; })      
    .attr("cy", function(d,i) {
        return yScale(d.measurement)+45; })    
    .attr("r", 14)
    .style("fill", function(d) {
        return colors(d.id);
    }) 

var circles_text_C2 = pointsLinesC2.selectAll(".text")
    .data(function(d){
        return d.values
    })
    .enter()
    .append("text")

    circles_text_C2.attr("x", function(d) { 
        return xScale(d.date)+100; })      
    .attr("y", function(d,i) {
        return yScaleSqrt(d.measurement)+50; })     
    .text(function(d) {
        return d.rank;})
    .style('fill', 'white');


legendC2 = svg_c_2.append("g")
.attr("id","legend-c-2")

legendC2.append("circle").attr("cx",width+200).attr("cy",300).attr("r", 20).style("fill", "black")
legendC2.append("text").attr("x", width+190).attr("y", 300).text("rank").style("font-size", "12px").style("fill","white")
legendC2.append("text").attr("x", width+150).attr("y", 330).text("BoardGameGeek Rank").style("font-size", "13px").style("fill","black")

});
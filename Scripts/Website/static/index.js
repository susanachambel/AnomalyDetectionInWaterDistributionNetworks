window.onload = function () {

    document.getElementById("demo").addEventListener("click", function () {

        console.log("hello")
        var data = {
            name: "Donald Duck",
            city: "Duckburg"
        }
        doWork(data)
    });
}

function doWork(data) {

    $.post("receiver", data, callbackFunc);
    // stop link reloading the page
    //event.preventDefault();
}

function callbackFunc(data, status) {


    var data_json = JSON.parse(data)

    data_jason['col 1']

    console.log(data_json)

    //alert("Data: " + data + "\nStatus: " + status);
}
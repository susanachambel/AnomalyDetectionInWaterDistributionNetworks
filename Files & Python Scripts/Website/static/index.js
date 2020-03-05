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
    console.log(data);
    alert("Data: " + data + "\nStatus: " + status);
}
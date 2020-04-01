var wmes = 0;
var sensor_list = 0;
var focus = 0;
var group = 0;
var type = 0;

window.onload = function () {

    var forms = document.getElementsByClassName('needs-validation');
    var validation = Array.prototype.filter.call(forms, function (form) {
        form.addEventListener('submit', function (event) {
            perform_manual_validation();
            if (form.checkValidity() === false) {
                event.preventDefault();
                event.stopPropagation();
                form.classList.add('was-validated');
            } else {
                submit();
                form.classList.remove('was-validated');
            };
        }, false);
    });

    document.getElementById("source").addEventListener("change", function () {
        update_form(this.value);
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            perform_manual_validation();
            form.checkValidity();
        };     
    });

    document.querySelectorAll("input[name=check-focus]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("focus", 1);
            };
            update_sensors(false, false);
        });
    });

    document.querySelectorAll("input[name=check-sensor-group]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("sensor-group", 1);
            };
            update_sensors(false, false);
        });
    });

    document.querySelectorAll("input[name=check-pairwise-comparisons]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("pairwise-comparisons", 1);
            };
        });
    });

    document.getElementById("sensor-type").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("sensor-type", 1);
        };
        update_sensors(false, false);
    });

    document.getElementById("sensor-name").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("sensor-name", 2);
        };
    });

    document.getElementById("calendar").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("calendar", 1);
        };
    });

    document.getElementById("correlation-type").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("correlation-type", 1);
        };
    });

    document.getElementById("correlation").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("correlation", 1);
        };
    });

    get_init_json();
};

function perform_manual_validation(){
    validate_check_box("focus", 1);
    validate_check_box("sensor-group", 1);
    validate_check_box("pairwise-comparisons", 1);
    validate_selection("sensor-type", 1);
    validate_selection("sensor-name", 2);
    validate_selection("calendar", 1);
    validate_selection("correlation-type", 1);
    validate_selection("correlation", 1);
}

function get_init_json() {
    console.log("hello");
    var data = {
        name: "json"
    };
    $.post("receiver", data, receive_init_json);
};

function receive_init_json(data, status) {
    var data_json = JSON.parse(data);

    wmes = {
        infraquinta: data_json['infraquinta'],
        barreiro: data_json['barreiro'],
        beja: data_json['beja']
    };

    init_form("infraquinta");
};


function update_form(wme) {

    $('#source').selectpicker('val', wme);

    switch (wme) {
        case "infraquinta":
            sensor_list = wmes.infraquinta;
            break;
        case "barreiro":
            sensor_list = wmes.barreiro;
            break;
        case "beja":
            sensor_list = wmes.beja;
            break;
    }

    focus = {
        real: false,
        simulated: false
    };
    group = {
        telemanagement: false,
        telemetry: false
    };
    type = {
        flow: false,
        pressure: false
    };

    console.log(sensor_list)

    for (var key in sensor_list) {
        if (!sensor_list.hasOwnProperty(key)) {
            continue;
        };

        var sensor = sensor_list[key];

        switch (sensor["focus"]) {
            case "real":
                focus.real = true;
                break;
            case "simulated":
                focus.simulated = true;
                break;
        };

        switch (sensor["group"]) {
            case "telemanagement":
                group.telemanagement = true;
                break;
            case "telemetry":
                group.telemetry = true;
                break;
        };

        switch (sensor["type"]) {
            case "flow":
                type.flow = true;
                break;
            case "pressure":
                type.pressure = true;
                break;
        };
    };

    update_selections_checkboxes();
    update_sensors(true, true);
}

function update_selections_checkboxes() {

    $("[value='real'][name='check-focus']").prop("checked", false);
    if (focus.real == false) {
        $("[value='real'][name='check-focus']").prop("disabled", true);
    } else {
        $("[value='real'][name='check-focus']").prop("disabled", false);
        $("[value='real'][name='check-focus']").prop("checked", true);
    }

    $("[value='simulated'][name='check-focus']").prop("checked", false);
    if (focus.simulated == false) {
        $("[value='simulated'][name='check-focus']").prop("disabled", true);
    } else {
        $("[value='simulated'][name='check-focus']").prop("disabled", false);
        $("[value='simulated'][name='check-focus']").prop("checked", true);
    }

    $("[value='telemetry'][name='check-sensor-group']").prop("checked", false);
    if (group.telemetry == false) {
        $("[value='telemetry'][name='check-sensor-group']").prop("disabled", true);
    } else {
        $("[value='telemetry'][name='check-sensor-group']").prop("disabled", false);
        $("[value='telemetry'][name='check-sensor-group']").prop("checked", true);
    }

    $("[value='telemanagement'][name='check-sensor-group']").prop("checked", false);
    if (group.telemanagement == false) {
        $("[value='telemanagement'][name='check-sensor-group']").prop("disabled", true);
    } else {
        $("[value='telemanagement'][name='check-sensor-group']").prop("disabled", false);
        $("[value='telemanagement'][name='check-sensor-group']").prop("checked", true);
    }

    $('#sensor-type').selectpicker('deselectAll');
    if (type.flow == false) {
        $('#sensor-type option[value=flow]').prop("disabled", true);
    } else {
        $('#sensor-type option[value=flow]').prop("disabled", false);
        $('#sensor-type option[value=flow]').prop("selected", true);
    }
    if (type.pressure == false) {
        $('#sensor-type option[value=pressure]').prop("disabled", true);
    } else {
        $('#sensor-type option[value=pressure]').prop("disabled", false);
        $('#sensor-type option[value=pressure]').prop("selected", true);
    }
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-type').selectpicker('render');
}

function update_sensors(delete_sensors, select_all) {

    if (delete_sensors) {
        $('#sensor-name').empty();
        $('#sensor-name').selectpicker('refresh');
        $('#sensor-name').selectpicker('render');
    }

    for (var key in sensor_list) {
        if (!sensor_list.hasOwnProperty(key)) {
            continue;
        };

        var sensor = sensor_list[key];
        var approved = true;

        switch (sensor["focus"]) {
            case "real":
                if ($("[name='check-focus'][value='real']:checked").length == 0) {
                    approved = false;
                }
                break;
            case "simulated":
                if ($("[name='check-focus'][value='simulated']:checked").length == 0) {
                    approved = false;
                }
                break;
        };

        switch (sensor["group"]) {
            case "telemanagement":
                if ($("[name='check-sensor-group'][value='telemanagement']:checked").length == 0) {
                    approved = false;
                }
                break;
            case "telemetry":
                if ($("[name='check-sensor-group'][value='telemetry']:checked").length == 0) {
                    approved = false;
                }
                break;
        };

        switch (sensor["type"]) {
            case "flow":
                if ($("#sensor-type option[value=flow]:selected").length == 0) {
                    approved = false;
                }
                break;
            case "pressure":
                if ($("#sensor-type option[value=pressure]:selected").length == 0) {
                    approved = false;
                }
                break;
        };

        if (approved && ($("#sensor-name option[value='" + key + "']").length == 0)) {
            var data_tokens = sensor["focus"] + " " + sensor["group"] + " " + sensor["type"];
            var data_subtext = dim(sensor["focus"]) + " • " + dim(sensor["group"]) + " • " + dim(sensor["type"]);
            var option = '<option value="' + key + '" data-subtext="' + data_subtext + '" data-tokens="' + data_tokens + '">' + sensor["name"] + '</option>';
            $('#sensor-name').append(option);
        } else {
            if ((!approved) && ($("#sensor-name option[value='" + key + "']").length != 0)) {
                $('#sensor-name').find('[value=' + key + ']').remove();
            }
        }
    };

    $('#sensor-name').selectpicker('refresh');
    $('#sensor-name').selectpicker('render');

    if (select_all) {
        $('#sensor-name').selectpicker('selectAll');
    }
}

function dim(name) {
    switch (name) {
        case "real":
            return "real";
        case "simulated":
            return "simu";
        case "telemanagement":
            return "tlmg";
        case "telemetry":
            return "tlmt";
        case "flow":
            return "flow";
        case "pressure":
            return "press";
    }
}

function submit() {
    var data = {
        name: "form"
    };

    deactivate_form();

    $.post("receiver", data, receive_data);
    event.preventDefault();
};

function receive_data(data, status) {

    var data_json = JSON.parse(data);

    lc_data = data_json["line_chart"];
    hm_data = data_json["heat_map"];

    create_line_chart(lc_data);
    create_heat_map();
    activate_form();
};

function validate_selection(field, length) {
    if ($('#' + field).val().length < length) {
        document.getElementById(field).setCustomValidity(' ');
        if (!document.getElementById("help-" + field)) {
            $("#div-" + field).append('<small id="help-' + field + '" class="text-muted">Select at least ' + length + ' </small>');
        };
    } else {
        document.getElementById(field).setCustomValidity('');
        if (document.getElementById("help-" + field)) {
            $("#help-" + field).remove();
        };
    }
};

function validate_check_box(field, length) {
    if ($("[name='check-" + field + "']:checked").length < length) {
        document.querySelectorAll("input[name=check-" + field + "]").forEach(box => {
            box.setCustomValidity(' ');
        });
        if (!document.getElementById("help-" + field)) {
            $("#div-" + field).append('<small id="help-' + field + '" class="text-muted">Check at least ' + length + ' </small>');
        };
    } else {
        document.querySelectorAll("input[name=check-" + field + "]").forEach(box => {
            box.setCustomValidity('');
        });
        if (document.getElementById("help-" + field)) {
            $("#help-" + field).remove();
        };
    };
};

/*
$('input[name="daterange"]').on('apply.daterangepicker', function (ev, picker) {
    console.log(picker.startDate.format('DD-MM-YYYY'));
    console.log(picker.endDate.format('DD-MM-YYYY'));
});
*/

function create_line_chart(lc_data) {
    var config = {
        responsive: true
    };

    var layout = {
        xaxis: {
            rangeslider: {}
        }
    };

    var data = [];

    for (sensor in lc_data) {
        
        var x = Object.keys(lc_data[sensor]);
        var y = Object.values(lc_data[sensor]);

        var trace = create_trace(sensor, x, y);
        data.push(trace);
    };

    Plotly.newPlot('line_chart', data, layout, config);
};

function create_trace(name, x, y) {
    var trace = {
        name: name,
        x: x,
        y: y,
        type: 'scatter'
    };
    return trace;
};


function create_heat_map() {

    var config = {
        responsive: true
    };

    var layout = {};

    var data = [{
        z: [
            [1, null, 30],
            [20, 1, 60],
            [30, 60, 1]
        ],
        x: ['Monday', 'Tuesday', 'Wednesday'],
        y: ['Morning', 'Afternoon', 'Evening'],
        colorscale: 'YIOrRd',
        reversescale: true,
        type: 'heatmap',
        hoverongaps: false
    }];

    Plotly.newPlot('heat_map', data, layout, config);
};


function init_form(wme) {
    update_form(wme);
    $('#correlation-type').selectpicker('refresh');
    $('#correlation-type').selectpicker('selectAll');
    $('#correlation').selectpicker('refresh');
    $('#correlation').selectpicker('selectAll');
    $('#calendar').selectpicker('refresh');
    $('#calendar').selectpicker('selectAll');
    $('#granularity').selectpicker('val', 'hours');
    $("[name='check-pairwise-comparisons'").prop("checked", true);
    $("[name='check-pca'").prop("checked", true);
    document.getElementById("check-default").checked = true;
    document.getElementById("granularity-value").value = 1;
    create_date_range_picker("01/06/2017", "08/06/2017", "01/01/2017", "31/12/2017", 7);
};


function create_date_range_picker(startDate, endDate, minDate, maxDate, maxSpan) {
    $(function () {
        $('input[name="daterange"]').daterangepicker({
            locale: {
                format: 'DD/MM/YYYY'
            },
            opens: 'center',
            "startDate": startDate,
            "endDate": endDate,
            "minDate": minDate,
            "maxDate": maxDate,
            "maxSpan": {
                "days": maxSpan
            }
        }, function (start, end, label) {
            console.log("A new date selection was made: " + start.format('YYYY-MM-DD') + ' to ' + end.format('YYYY-MM-DD'));
        });
    });
};


function deactivate_form() {
    $('#run-btn').prop("disabled", true);
    $("#run-btn").html(
        `<span id="btn-spinner" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...`
    );

    $('#source').prop("disabled", true);
    $('#source').selectpicker('refresh');
    $('#form-target input[type=checkbox]').prop("disabled", true);
    $('#sensor-type').prop("disabled", true);
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-name').prop("disabled", true);
    $('#sensor-name').selectpicker('refresh');
    $('#calendar').prop("disabled", true);
    $('#calendar').selectpicker('refresh');
    $('#form-target input[type=text]').prop("disabled", true);
    $('#granularity').prop("disabled", true);
    $('#granularity').selectpicker('refresh');
    $('#form-target input[type=number]').prop("disabled", true);

    $('#form-target input[type=radio]').prop("disabled", true);
    $('#correlation-type').prop('disabled', true);
    $('#correlation-type').selectpicker('refresh');
    $('#correlation').prop('disabled', true);
    $('#correlation').selectpicker('refresh');
};

function activate_form() {
    $('#source').prop("disabled", false);
    $('#source').selectpicker('refresh');
    $('#form-target input[type=checkbox]').prop("disabled", false);
    $('#sensor-type').prop("disabled", false);
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-name').prop("disabled", false);
    $('#sensor-name').selectpicker('refresh');
    $('#calendar').prop("disabled", false);
    $('#calendar').selectpicker('refresh');
    $('#form-target input[type=text]').prop("disabled", false);
    $('#granularity').prop("disabled", false);
    $('#granularity').selectpicker('refresh');
    $('#form-target input[type=number]').prop("disabled", false);

    $('#form-target input[type=radio]').prop("disabled", false);
    $('#correlation-type').prop('disabled', false);
    $('#correlation-type').selectpicker('refresh');
    $('#correlation').prop('disabled', false);
    $('#correlation').selectpicker('refresh');

    if (!focus.real) {
        $("[value='real'][name='check-focus']").prop("disabled", true);
    }

    if (!focus.simulated) {
        $("[value='simulated'][name='check-focus']").prop("disabled", true);
    }

    if (!group.telemanagement) {
        $("[value='telemanagement'][name='check-sensor-group']").prop("disabled", true);
    }

    if (!group.telemetry) {
        $("[value='telemetry'][name='check-sensor-group']").prop("disabled", true);
    }

    $("#btn-spinner").remove();
    $("#run-btn").html(
        `Run Query`
    );
    $("#run-btn").prop("disabled", false);
};